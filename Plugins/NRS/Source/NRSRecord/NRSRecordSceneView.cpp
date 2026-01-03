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

static TAutoConsoleVariable<int32> CVarNRSRecord(
	TEXT("r.NRS.Record"),
	0,
	TEXT("Enable NRS Record\n0: off, 1: on"),
	ECVF_Default | ECVF_RenderThreadSafe);

uint64 NRSRecordSceneViewExtension::NRSReadbackFrameId = 0;

NRSRecordSceneViewExtension::NRSRecordSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister),
	SceneColorReadback(TEXT("NRS_SceneColorReadback")),
	SceneDepthReadback(TEXT("NRS_SceneDepthReadback")),
	CameraMotionReadback(TEXT("NRS_CameraMotionReadback"))
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

		MotionVectorTexture = GraphBuilder.CreateTexture(OutputDesc, TEXT("NRSRecord_CameraMotion"));
	}
	else
	{
		MotionVectorTexture = GraphBuilder.RegisterExternalTexture(MotionVectorRT);
	}

	AddMotionGeneration(GraphBuilder, InView, SceneDepthTexture, VelocityTexture, MotionVectorTexture);

	if (CVarNRSRecord.GetValueOnAnyThread() != 0)
	{
		RecordBuffer(GraphBuilder, InView, SceneColorTexture, SceneColorReadback, TEXT("SceneColor"));
		RecordDepthBuffer(GraphBuilder, InView, SceneDepthTexture, SceneDepthReadback, TEXT("SceneDepth"));
		RecordBuffer(GraphBuilder, InView, MotionVectorTexture, CameraMotionReadback, TEXT("CameraMotion"));
	}

	GraphBuilder.QueueTextureExtraction(MotionVectorTexture, &MotionVectorRT);
}

void NRSRecordSceneViewExtension::AddMotionGeneration(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SceneDepthTexture,
		FRDGTextureRef SceneVelocityTexture,
		FRDGTextureRef MotionVectorTexture)
{
	const FViewInfo& View = (FViewInfo &)InView;

	UE_LOG(LogTemp, Log, TEXT("AddMotionGeneration: %d x %d"), ViewSize.X, ViewSize.Y);

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
