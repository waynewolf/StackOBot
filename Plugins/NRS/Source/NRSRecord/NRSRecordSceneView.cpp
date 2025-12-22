#include "NRSRecordSceneView.h"
#include "NRSRecordShaders.h"

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

namespace
{
	struct NRSReadbackState
	{
		TUniquePtr<FRHIGPUTextureReadback> Readback;
		FIntPoint Size = FIntPoint::ZeroValue;
		EPixelFormat Format = PF_Unknown;
		uint64 FrameId = 0;

		explicit NRSReadbackState(const TCHAR* Name)
			: Readback(MakeUnique<FRHIGPUTextureReadback>(Name))
		{
		}
	};

	static uint64 GNRSReadbackFrameId = 0;

	static void SaveReadbackIfReady(NRSReadbackState& State, const FString& Label)
	{
		if (!State.Readback || !State.Readback->IsReady())
		{
			return;
		}

		int32 RowPitchInPixels = 0;
		void* Data = State.Readback->Lock(RowPitchInPixels);
		if (!Data)
		{
			State.Readback->Unlock();
			return;
		}

		const int32 BytesPerPixel = GPixelFormats[State.Format].BlockBytes;
		if (BytesPerPixel <= 0 || State.Size.X <= 0 || State.Size.Y <= 0)
		{
			State.Readback->Unlock();
			return;
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
			TEXT("%s_%llu_%dx%d.bin"),
			*Label,
			State.FrameId,
			State.Size.X,
			State.Size.Y);
		FFileHelper::SaveArrayToFile(Output, *OutputPath);
	}
}

IMPLEMENT_GLOBAL_SHADER(NRSMotionGenCS, "/Plugin/NRS/MotionGen.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(NRSMotionVizPS, "/Plugin/NRS/MotionViz.usf", "MainPS", SF_Pixel);
IMPLEMENT_GLOBAL_SHADER(NRSVisualizeXPS, "/Plugin/NRS/VisualizeX.usf", "MainPS", SF_Pixel);

static TAutoConsoleVariable<int32> CVarNRSRecordBuffers(
	TEXT("r.NRS.RecordBuffers"),
	1,
	TEXT("Enable NRS RecordBuffers pass.\n0: off, 1: on"),
	ECVF_Default);

NRSRecordSceneViewExtension::NRSRecordSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister)
{
}

void NRSRecordSceneViewExtension::PrePostProcessPass_RenderThread(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	const FPostProcessingInputs& Inputs)
{
	const FViewInfo& View = (FViewInfo &)InView;
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
		CachedMotionVectorTexture = MotionVectorTexture;
	}
	else
	{
		MotionVectorTexture = GraphBuilder.RegisterExternalTexture(MotionVectorRT);
	}

	AddMotionGeneration(GraphBuilder, InView, SceneDepthTexture, VelocityTexture, MotionVectorTexture);

	const bool bIsGameView = View.bIsGameView || (View.Family && View.Family->EngineShowFlags.Game);
	if (bIsGameView)
	{
		AddMotionVisualization(GraphBuilder, InView, SceneColorTexture, SceneDepthTexture, MotionVectorTexture);

		UE_LOG(
			LogTemp,
			Log,
			TEXT("View.AntiAliasingMethod: %d, View.bCameraCut: %d, View.bPrevTransformsReset: %d"),
			(int)View.AntiAliasingMethod,
			(int)View.bCameraCut,
			(int)View.bPrevTransformsReset);

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
	}

	GraphBuilder.QueueTextureExtraction(MotionVectorTexture, &MotionVectorRT);

	CachedPPInputs = Inputs;
}

void NRSRecordSceneViewExtension::SubscribeToPostProcessingPass(
	EPostProcessingPass Pass,
	const FSceneView& InView,
	FPostProcessingPassDelegateArray& InOutPassCallbacks,
	bool bIsPassEnabled)
{
	const FViewInfo& View = (FViewInfo &)InView;

	const bool bIsGameView = View.bIsGameView || (View.Family && View.Family->EngineShowFlags.Game);

	// Cannot run into Tonemap, ReplacingTonemapper, MotionBlur
	if (Pass == EPostProcessingPass::BeforeDOF && bIsPassEnabled && bIsGameView)
	{
		UE_LOG(LogTemp, Log, TEXT("BeforeDOF"));
		InOutPassCallbacks.Add(FPostProcessingPassDelegate::CreateRaw(this, &NRSRecordSceneViewExtension::InPostProcessChain));
	}
}

FScreenPassTexture NRSRecordSceneViewExtension::InPostProcessChain(
	FRDGBuilder& GraphBuilder,
	const FSceneView& View,
	const FPostProcessMaterialInputs& Inputs)
{
	RDG_EVENT_SCOPE(GraphBuilder, "InPostProcessChain");

	FScreenPassTexture SceneColorScreenPassTexture(Inputs.GetInput(EPostProcessMaterialInput::SceneColor));
	FScreenPassTexture TranslucencyScreenPassTexture(Inputs.GetInput(EPostProcessMaterialInput::SeparateTranslucency));

	AddXVisualization(GraphBuilder, View, TranslucencyScreenPassTexture.Texture, SceneColorScreenPassTexture.Texture);

	FRDGTextureRef SceneColorTexture = (*CachedPPInputs.SceneTextures)->SceneColorTexture;
	FRDGTextureRef SceneDepthTexture = (*CachedPPInputs.SceneTextures)->SceneDepthTexture;

	FRDGTextureRef GBufferA = (*CachedPPInputs.SceneTextures)->GBufferATexture;
	FRDGTextureRef GBufferB = (*CachedPPInputs.SceneTextures)->GBufferBTexture;
	FRDGTextureRef GBufferC = (*CachedPPInputs.SceneTextures)->GBufferCTexture;

	if (CVarNRSRecordBuffers.GetValueOnRenderThread() != 0)
	{
		RecordBuffers(
			GraphBuilder,
			View,
			SceneColorTexture,
			SceneDepthTexture,
			CachedMotionVectorTexture,
			TranslucencyScreenPassTexture.Texture,
			GBufferA,
			GBufferB,
			GBufferC);
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
	const FIntPoint ViewSize = View.ViewRect.Size();

	UE_LOG(LogTemp, Log, TEXT("ViewRect: %d x %d, Min: %d x %d"), ViewSize.X, ViewSize.Y, View.ViewRect.Min.X, View.ViewRect.Min.Y);

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

void NRSRecordSceneViewExtension::RecordBuffers(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	FRDGTextureRef SceneColorTexture,
	FRDGTextureRef SceneDepthTexture,
	FRDGTextureRef MotionVectorTexture,
	FRDGTextureRef TranslucencyTexture,
	FRDGTextureRef GBufferATexture,
	FRDGTextureRef GBufferBTexture,
	FRDGTextureRef GBufferCTexture)
{
	const FViewInfo& View = (FViewInfo &)InView;

	static NRSReadbackState SceneColorReadback(TEXT("NRS_SceneColorReadback"));
	//static NRSReadbackState SceneDepthReadback(TEXT("NRS_SceneDepthReadback"));
	//static NRSReadbackState MotionReadback(TEXT("NRS_MotionReadback"));
	static NRSReadbackState GBufferAReadback(TEXT("NRS_GBufferAReadback"));
	static NRSReadbackState GBufferBReadback(TEXT("NRS_GBufferBReadback"));
	static NRSReadbackState GBufferCReadback(TEXT("NRS_GBufferCReadback"));

	SaveReadbackIfReady(SceneColorReadback, TEXT("SceneColor"));
	//SaveReadbackIfReady(SceneDepthReadback, TEXT("SceneDepth"));
	//SaveReadbackIfReady(MotionReadback, TEXT("Motion"));
	SaveReadbackIfReady(GBufferAReadback, TEXT("GBufferA"));
	SaveReadbackIfReady(GBufferBReadback, TEXT("GBufferB"));
	SaveReadbackIfReady(GBufferCReadback, TEXT("GBufferC"));

	++GNRSReadbackFrameId;

	if (SceneColorTexture)
	{
		SceneColorReadback.Size = SceneColorTexture->Desc.Extent;
		SceneColorReadback.Format = SceneColorTexture->Desc.Format;
		SceneColorReadback.FrameId = GNRSReadbackFrameId;
		AddEnqueueCopyPass(GraphBuilder, SceneColorReadback.Readback.Get(), SceneColorTexture);
	}
	// if (SceneDepthTexture)
	// {
	// 	SceneDepthReadback.Size = SceneDepthTexture->Desc.Extent;
	// 	SceneDepthReadback.Format = SceneDepthTexture->Desc.Format;
	// 	SceneDepthReadback.FrameId = GNRSReadbackFrameId;
	// 	AddEnqueueCopyPass(GraphBuilder, SceneDepthReadback.Readback.Get(), SceneDepthTexture);
	// }
	// if (MotionVectorTexture)
	// {
	// 	MotionReadback.Size = MotionVectorTexture->Desc.Extent;
	// 	MotionReadback.Format = MotionVectorTexture->Desc.Format;
	// 	MotionReadback.FrameId = GNRSReadbackFrameId;
	// 	AddEnqueueCopyPass(GraphBuilder, MotionReadback.Readback.Get(), MotionVectorTexture);
	// }
	if (GBufferATexture)
	{
		GBufferAReadback.Size = GBufferATexture->Desc.Extent;
		GBufferAReadback.Format = GBufferATexture->Desc.Format;
		GBufferAReadback.FrameId = GNRSReadbackFrameId;
		AddEnqueueCopyPass(GraphBuilder, GBufferAReadback.Readback.Get(), GBufferATexture);
	}
	if (GBufferBTexture)
	{
		GBufferBReadback.Size = GBufferBTexture->Desc.Extent;
		GBufferBReadback.Format = GBufferBTexture->Desc.Format;
		GBufferBReadback.FrameId = GNRSReadbackFrameId;
		AddEnqueueCopyPass(GraphBuilder, GBufferBReadback.Readback.Get(), GBufferBTexture);
	}
	if (GBufferCTexture)
	{
		GBufferCReadback.Size = GBufferCTexture->Desc.Extent;
		GBufferCReadback.Format = GBufferCTexture->Desc.Format;
		GBufferCReadback.FrameId = GNRSReadbackFrameId;
		AddEnqueueCopyPass(GraphBuilder, GBufferCReadback.Readback.Get(), GBufferCTexture);
	}

	const FScreenPassTexture OutputScreenPassTexture(SceneColorTexture);
	const FScreenPassRenderTarget OutputRT(OutputScreenPassTexture, ERenderTargetLoadAction::ENoAction);
	FScreenPassTextureViewport OutputViewport(OutputRT);

	NRSVisualizeXPS::FParameters* PassParameters = GraphBuilder.AllocParameters<NRSVisualizeXPS::FParameters>();
	PassParameters->InputTexture = GBufferATexture;
	PassParameters->InputTextureSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->RenderTargets[0] = OutputRT.GetRenderTargetBinding();

	TShaderMapRef<NRSVisualizeXPS> PixelShader(View.ShaderMap);
	AddDrawScreenPass(
		GraphBuilder,
		RDG_EVENT_NAME("RecordBuffers"),
		FScreenPassViewInfo(InView),
		OutputViewport,
		OutputViewport,
		PixelShader,
		PassParameters
	);
}
