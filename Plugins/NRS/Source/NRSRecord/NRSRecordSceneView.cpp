#include "NRSRecordSceneView.h"
#include "NRSRecordShaders.h"

#include "PostProcess/PostProcessInputs.h"
#include "PostProcess/PostProcessMaterialInputs.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "ScreenPass.h"
#include "SceneView.h"
#include "SceneRendering.h"
#include "ScenePrivate.h"
#include "HAL/IConsoleManager.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "PixelFormat.h"
#include "RHICommandList.h"
#include "RHIGPUReadback.h"


IMPLEMENT_GLOBAL_SHADER(NRSMotionGenCS, "/Plugin/NRS/MotionGen.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(NRSDepthToFloatCS, "/Plugin/NRS/DepthToFloat.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(NRSMotionVizPS, "/Plugin/NRS/MotionViz.usf", "MainPS", SF_Pixel);
IMPLEMENT_GLOBAL_SHADER(NRSVisualizeXPS, "/Plugin/NRS/VisualizeX.usf", "MainPS", SF_Pixel);

static TAutoConsoleVariable<int32> CVarNRSVisualizeX(
	TEXT("r.NRS.VisualizeX"),
	0,
	TEXT("Enable NRS VisualizeX pass.\n0: off, 1: on"),
	ECVF_Default | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNRSRecordBasic(
	TEXT("r.NRS.RecordBasic"),
	0,
	TEXT("Enable NRS RecordBasic\n0: off, 1: on"),
	ECVF_Default | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNRSRecordMotion(
	TEXT("r.NRS.RecordMotion"),
	0,
	TEXT("Enable NRS RecordMotion\n0: off, 1: on"),
	ECVF_Default | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNRSRecordTranslucency(
	TEXT("r.NRS.RecordTranslucency"),
	0,
	TEXT("Enable NRS RecordTranslucency\n0: off, 1: on"),
	ECVF_Default | ECVF_RenderThreadSafe);

static TAutoConsoleVariable<int32> CVarNRSRecordAll(
	TEXT("r.NRS.RecordAll"),
	0,
	TEXT("Enable NRS RecordAll pass.\n0: off, 1: on"),
	ECVF_Default | ECVF_RenderThreadSafe);

uint64 NRSRecordSceneViewExtension::NRSReadbackFrameId = 0;

NRSRecordSceneViewExtension::NRSRecordSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister),
	SceneColorReadback(TEXT("NRS_SceneColorReadback")),
	SceneDepthReadback(TEXT("NRS_SceneDepthReadback")),
	MotionVectorReadback(TEXT("NRS_MotionVectorReadback")),
	TranslucencyReadback(TEXT("NRS_TranslucencyReadback")),
	GBufferCReadback(TEXT("NRS_GBufferCReadback"))
{
}

void NRSRecordSceneViewExtension::PrePostProcessPass_RenderThread(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	const FPostProcessingInputs& Inputs)
{
	const bool bIsGameView = InView.bIsGameView || (InView.Family && InView.Family->EngineShowFlags.Game);
	if (!bIsGameView)
	{
		return;
	}

	const FViewInfo& View = (FViewInfo &)InView;
	ViewSize = View.ViewRect.Size();
	UE_LOG(LogTemp, Log, TEXT("Game ViewRect: %d x %d, Min: %d x %d"), ViewSize.X, ViewSize.Y, View.ViewRect.Min.X, View.ViewRect.Min.Y);

	NRSReadbackFrameId++;
	
	FRDGTextureRef SceneColorTexture = (*Inputs.SceneTextures)->SceneColorTexture;
	FRDGTextureRef SceneDepthTexture = (*Inputs.SceneTextures)->SceneDepthTexture;
	FRDGTextureRef VelocityTexture = (*Inputs.SceneTextures)->GBufferVelocityTexture;

	FRDGTextureRef MotionVectorTexture;

	if (!MotionVectorRT.IsValid() ||
		MotionVectorRT->GetDesc().Extent.X != SceneDepthTexture->Desc.Extent.X ||
		MotionVectorRT->GetDesc().Extent.Y != SceneDepthTexture->Desc.Extent.Y ||
		MotionVectorRT->GetDesc().Format != PF_G16R16F)
	{
		FRDGTextureDesc OutputDesc = FRDGTextureDesc::Create2D(
			FIntPoint(SceneDepthTexture->Desc.Extent.X, SceneDepthTexture->Desc.Extent.Y),
			PF_G16R16F,
			FClearValueBinding::None,
			TexCreate_UAV | TexCreate_ShaderResource | TexCreate_RenderTargetable
		);

		MotionVectorTexture = GraphBuilder.CreateTexture(OutputDesc, TEXT("NRSRecord_MotionVector"));
	}
	else
	{
		MotionVectorTexture = GraphBuilder.RegisterExternalTexture(MotionVectorRT);
	}

	AddMotionGeneration(GraphBuilder, InView, SceneDepthTexture, VelocityTexture, MotionVectorTexture);

	if (CVarNRSRecordBasic.GetValueOnAnyThread() != 0 || CVarNRSRecordAll.GetValueOnAnyThread() != 0)
	{
		RecordBuffer(GraphBuilder, InView, SceneColorTexture, SceneColorReadback, TEXT("SceneColor"));
		RecordDepthBuffer(GraphBuilder, InView, SceneDepthTexture, SceneDepthReadback, TEXT("SceneDepth"));
	}

	if (CVarNRSRecordMotion.GetValueOnAnyThread() != 0 || CVarNRSRecordAll.GetValueOnAnyThread() != 0)
	{
		RecordBuffer(GraphBuilder, InView, MotionVectorTexture, MotionVectorReadback, TEXT("MotionVector"));
	}

	if (CVarNRSVisualizeX.GetValueOnAnyThread() != 0)
	{
		AddMotionVisualization(GraphBuilder, InView, SceneColorTexture, SceneDepthTexture, MotionVectorTexture);
	}

	if (View.ViewState)
	{
		const FTemporalAAHistory& TAAHistory = View.ViewState->PrevFrameViewInfo.TemporalAAHistory;
		if (TAAHistory.IsValid())
		{
			UE_LOG(LogTemp, Log, TEXT("Has TemporalAAHistory"));
			// Not called each frame, why?
			//FRDGTextureRef HistoryTexture = GraphBuilder.RegisterExternalTexture(TAAHistory.RT[0]);
			//AddXVisualization(GraphBuilder, InView, HistoryTexture, SceneColorTexture);
		}
	}

	GraphBuilder.QueueTextureExtraction(MotionVectorTexture, &MotionVectorRT);
}

void NRSRecordSceneViewExtension::SubscribeToPostProcessingPass(
	EPostProcessingPass Pass,
	const FSceneView& InView,
	FPostProcessingPassDelegateArray& InOutPassCallbacks,
	bool bIsPassEnabled)
{
	const bool bIsGameView = InView.bIsGameView || (InView.Family && InView.Family->EngineShowFlags.Game);
	if (!bIsGameView)
	{
		return;
	}

	// Cannot run into Tonemap, ReplacingTonemapper, MotionBlur
	if (Pass == EPostProcessingPass::BeforeDOF && bIsPassEnabled)
	{
		UE_LOG(LogTemp, Log, TEXT("BeforeDOF"));
		InOutPassCallbacks.Add(FPostProcessingPassDelegate::CreateRaw(this, &NRSRecordSceneViewExtension::InPostProcessChain));
	}
}

FScreenPassTexture NRSRecordSceneViewExtension::InPostProcessChain(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	const FPostProcessMaterialInputs& Inputs)
{
	RDG_EVENT_SCOPE(GraphBuilder, "InPostProcessChain");

	FScreenPassTexture SceneColorScreenPassTexture(Inputs.GetInput(EPostProcessMaterialInput::SceneColor));
	FScreenPassTexture TranslucencyScreenPassTexture(Inputs.GetInput(EPostProcessMaterialInput::SeparateTranslucency));

	if (CVarNRSVisualizeX.GetValueOnAnyThread() != 0)
	{
		AddXVisualization(GraphBuilder, InView, TranslucencyScreenPassTexture.Texture, SceneColorScreenPassTexture.Texture);
	}

	if (CVarNRSRecordTranslucency.GetValueOnAnyThread() != 0 || CVarNRSRecordAll.GetValueOnAnyThread() != 0)
	{
		FRDGTextureRef TranslucencyTexture = TranslucencyScreenPassTexture.Texture;
		RecordBuffer(GraphBuilder, InView, TranslucencyTexture, TranslucencyReadback, TEXT("Translucency"));
	}

	return SceneColorScreenPassTexture;
}

void NRSRecordSceneViewExtension::AddMotionGeneration(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SceneDepthTexture,
		FRDGTextureRef SceneVelocityTexture,
		FRDGTextureRef MotionVectorTexture)
{
	const FViewInfo& View = (FViewInfo &)InView;

	FRDGTextureSRVRef SceneDepthSRV = GraphBuilder.CreateSRV(SceneDepthTexture);
	FRDGTextureSRVRef SceneVelocitySRV = GraphBuilder.CreateSRV(SceneVelocityTexture);
	FRDGTextureUAVRef MotionVectorUAV = GraphBuilder.CreateUAV(MotionVectorTexture);

	AddClearUAVPass(GraphBuilder, MotionVectorUAV, FVector4(0, 0, 0, 0));

	NRSMotionGenCS::FParameters* PassParameters = GraphBuilder.AllocParameters<NRSMotionGenCS::FParameters>();
	PassParameters->DepthTexture = SceneDepthTexture;
	PassParameters->InputDepth = SceneDepthSRV;
	PassParameters->InputVelocity = SceneVelocitySRV;
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->OutputTexture = MotionVectorUAV;

	TShaderMapRef<NRSMotionGenCS> ComputeShader(View.ShaderMap);
	FComputeShaderUtils::AddPass(
		GraphBuilder,
		RDG_EVENT_NAME("MotionGen"),
		ComputeShader,
		PassParameters,
		FComputeShaderUtils::GetGroupCount(
			FIntVector(ViewSize.X, ViewSize.Y, 1),
			FIntVector(NRSMotionGenCS::ThreadgroupSizeX, NRSMotionGenCS::ThreadgroupSizeY, NRSMotionGenCS::ThreadgroupSizeZ))
	);
}

void NRSRecordSceneViewExtension::AddMotionVisualization(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	FRDGTextureRef SceneColorTexture,
	FRDGTextureRef SceneDepthTexture,
	FRDGTextureRef MotionVectorTexture)
{
	const FViewInfo& View = (FViewInfo &)InView;

	const FScreenPassTexture SceneColor(SceneColorTexture, View.ViewRect);
	const FScreenPassRenderTarget Output(SceneColor, ERenderTargetLoadAction::ELoad);
	FScreenPassTextureViewport OutputViewport(Output);

	FRDGTextureRef SceneColorCopy = GraphBuilder.CreateTexture(SceneColorTexture->Desc, TEXT("NRSRecord_SceneColorCopy"));
	AddCopyTexturePass(GraphBuilder, SceneColorTexture, SceneColorCopy);

	NRSMotionVizPS::FParameters* PassParameters = GraphBuilder.AllocParameters<NRSMotionVizPS::FParameters>();
	PassParameters->InputMotion = MotionVectorTexture;
	PassParameters->InputMotionSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->InputColor = SceneColorCopy;
	PassParameters->InputColorSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->RenderTargets[0] = Output.GetRenderTargetBinding();

	TShaderMapRef<NRSMotionVizPS> PixelShader(View.ShaderMap);
	AddDrawScreenPass(
		GraphBuilder,
		RDG_EVENT_NAME("MotionViz"),
		FScreenPassViewInfo(InView),
		OutputViewport,
		OutputViewport,
		PixelShader,
		PassParameters
	);
}

void NRSRecordSceneViewExtension::AddXVisualization(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	FRDGTextureRef TranslucencyTexture,
	FRDGTextureRef OutputTexture)
{
	const FViewInfo& View = (FViewInfo &)InView;

	const FScreenPassTexture TranslucencyScreenPassTexture(TranslucencyTexture);
	const FScreenPassTexture OutputScreenPassTexture(OutputTexture);

	const FScreenPassRenderTarget OutputRT(OutputScreenPassTexture, ERenderTargetLoadAction::ENoAction);
	FScreenPassTextureViewport OutputViewport(OutputRT);

	NRSVisualizeXPS::FParameters* PassParameters = GraphBuilder.AllocParameters<NRSVisualizeXPS::FParameters>();
	PassParameters->InputTexture = TranslucencyScreenPassTexture.Texture;
	PassParameters->InputTextureSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->RenderTargets[0] = OutputRT.GetRenderTargetBinding();

	TShaderMapRef<NRSVisualizeXPS> PixelShader(View.ShaderMap);
	AddDrawScreenPass(
		GraphBuilder,
		RDG_EVENT_NAME("VisualizeX"),
		FScreenPassViewInfo(InView),
		OutputViewport,
		OutputViewport,
		PixelShader,
		PassParameters
	);
}

void NRSRecordSceneViewExtension::RecordBuffer(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	FRDGTextureRef InTexture,
	NRSReadbackState& Readback,
	const FString& Label)
{
	if (InTexture == nullptr)
	{
		return;
	}

	const FIntPoint NewSize = InTexture->Desc.Extent;
	const EPixelFormat NewFormat = InTexture->Desc.Format;
	const bool bSizeOrFormatChanged =
		(Readback.PendingSize != NewSize) || (Readback.PendingFormat != NewFormat);
	if (bSizeOrFormatChanged)
	{
		Readback.bNeedsReset = true;
	}

	if (Readback.bPendingCopy)
	{
		if (!SaveReadbackIfReady(Readback, Label))
		{
			return;
		}
		Readback.bPendingCopy = false;
	}

	if (Readback.bNeedsReset)
	{
		Readback.Readback = MakeUnique<FRHIGPUTextureReadback>(Readback.Name);
		Readback.bNeedsReset = false;
	}

	Readback.PendingSize = NewSize;
	Readback.PendingFormat = NewFormat;
	Readback.Size = NewSize;
	Readback.Format = NewFormat;
	Readback.FrameId = NRSReadbackFrameId;
	AddEnqueueCopyPass(GraphBuilder, Readback.Readback.Get(), InTexture);
	Readback.bPendingCopy = true;
}

void NRSRecordSceneViewExtension::RecordDepthBuffer(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	FRDGTextureRef InTexture,
	NRSReadbackState& Readback,
	const FString& Label)
{
	if (InTexture == nullptr)
	{
		return;
	}

	const EPixelFormat Format = InTexture->Desc.Format;
	const bool bIsDepthOrStencil = IsDepthOrStencilFormat(Format);
	if (!bIsDepthOrStencil)
	{
		RecordBuffer(GraphBuilder, InView, InTexture, Readback, Label);
		return;
	}

	const FViewInfo& View = (FViewInfo &)InView;
	const FIntPoint TextureSize = InTexture->Desc.Extent;

	const FRDGTextureDesc OutputDesc = FRDGTextureDesc::Create2D(
		TextureSize,
		PF_R32_FLOAT,
		FClearValueBinding::None,
		TexCreate_UAV | TexCreate_ShaderResource
	);

	FRDGTextureRef DepthFloatTexture = GraphBuilder.CreateTexture(OutputDesc, TEXT("NRSRecord_DepthFloat"));
	FRDGTextureSRVRef DepthSRV = GraphBuilder.CreateSRV(InTexture);
	FRDGTextureUAVRef DepthUAV = GraphBuilder.CreateUAV(DepthFloatTexture);

	NRSDepthToFloatCS::FParameters* PassParameters = GraphBuilder.AllocParameters<NRSDepthToFloatCS::FParameters>();
	PassParameters->DepthTexture = InTexture;
	PassParameters->InputDepth = DepthSRV;
	PassParameters->OutputDepth = DepthUAV;
	PassParameters->TextureSize = TextureSize;

	TShaderMapRef<NRSDepthToFloatCS> ComputeShader(View.ShaderMap);
	FComputeShaderUtils::AddPass(
		GraphBuilder,
		RDG_EVENT_NAME("DepthToFloat"),
		ComputeShader,
		PassParameters,
		FComputeShaderUtils::GetGroupCount(
			FIntVector(TextureSize.X, TextureSize.Y, 1),
			FIntVector(NRSDepthToFloatCS::ThreadgroupSizeX, NRSDepthToFloatCS::ThreadgroupSizeY, NRSDepthToFloatCS::ThreadgroupSizeZ))
	);

	RecordBuffer(GraphBuilder, InView, DepthFloatTexture, Readback, Label);
}

bool NRSRecordSceneViewExtension::SaveReadbackIfReady(NRSReadbackState& State, const FString& Label)
{
	if (!State.Readback || !State.Readback->IsReady())
	{
		return false;
	}

	int32 RowPitchInPixels = 0;
	void* Data = State.Readback->Lock(RowPitchInPixels);
	if (!Data)
	{
		State.Readback->Unlock();
		return false;
	}

	const int32 BytesPerPixel = GPixelFormats[State.Format].BlockBytes;
	if (BytesPerPixel <= 0 || State.Size.X <= 0 || State.Size.Y <= 0)
	{
		State.Readback->Unlock();
		return false;
	}
	const int32 RowBytes = State.Size.X * BytesPerPixel;
	TArray<uint8> Output;
	Output.SetNumUninitialized(State.Size.X * State.Size.Y * BytesPerPixel);

	const uint8* Src = static_cast<const uint8*>(Data);
	uint8* Dst = Output.GetData();
	for (int32 Y = 0; Y < State.Size.Y; ++Y)
	{
		FMemory::Memcpy(Dst + Y * RowBytes, Src + Y * RowPitchInPixels * BytesPerPixel, RowBytes);
	}

	State.Readback->Unlock();

	const FString OutputDir = FPaths::ProjectSavedDir() / TEXT("NRSRecord");
	IFileManager::Get().MakeDirectory(*OutputDir, true);
	const FString OutputPath = OutputDir / FString::Printf(
		TEXT("%06llu_%s_%dx%d_in_%dx%d.data"),
		State.FrameId,
		*Label,
		ViewSize.Y,
		ViewSize.X,
		State.Size.Y,
		State.Size.X);
	FFileHelper::SaveArrayToFile(Output, *OutputPath);

	return true;
}
