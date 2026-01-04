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
#include "Math/Vector2D.h"
#include "RHICommandList.h"
#include "RHIGPUReadback.h"


IMPLEMENT_GLOBAL_SHADER(NRSCopyColorPS, "/Plugin/NRS/NRSCopyColor.usf", "MainPS", SF_Pixel);
IMPLEMENT_GLOBAL_SHADER(NRSCameraMotionPS, "/Plugin/NRS/NRSCameraMotion.usf", "MainPS", SF_Pixel);

static TAutoConsoleVariable<int32> CVarNRSRecord(
	TEXT("r.NRS.Record"),
	1,
	TEXT("Enable NRS Record\n0: off, 1: on"),
	ECVF_Default | ECVF_RenderThreadSafe);

uint64 NRSRecordSceneViewExtension::NRSReadbackFrameId = 0;
int NRSRecordSceneViewExtension::DestViewSizeX = 448;
int NRSRecordSceneViewExtension::DestViewSizeY = 352;

NRSRecordSceneViewExtension::NRSRecordSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister),
	SceneColorReadback(TEXT("NRS_SceneColorReadback")),
	SceneDepthReadback(TEXT("NRS_SceneDepthReadback")),
	CameraMotionReadback(TEXT("NRS_CameraMotionReadback"))
{
}

void NRSRecordSceneViewExtension::PreRenderView_RenderThread(FRDGBuilder& GraphBuilder, FSceneView& InView)
{
	const bool bIsGameView = InView.bIsGameView || (InView.Family && InView.Family->EngineShowFlags.Game);
	if (!bIsGameView)
	{
		return;
	}

	FViewInfo& View = (FViewInfo &)InView;
	const FIntRect OldRect = View.ViewRect;

	if (OldRect.Width() < DestViewSizeX || OldRect.Height() < DestViewSizeY)
	{
		UE_LOG(LogTemp, Log, TEXT("ViewRect is smaller than NRS destination size, not resizing."));
		return;
	}

	UE_LOG(LogTemp, Log, TEXT("Resizing ViewRect from Min(%d, %d) Max(%d, %d) -> %d x %d"),
		OldRect.Min.X, OldRect.Min.Y, OldRect.Max.X, OldRect.Max.Y, DestViewSizeX,  DestViewSizeY);

	FIntRect NewRect(0, 0, NRSRecordSceneViewExtension::DestViewSizeX, NRSRecordSceneViewExtension::DestViewSizeY);
	View.ViewRect = NewRect;
    View.UnconstrainedViewRect = NewRect;
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
	SourceViewSize = View.ViewRect.Size();

	UE_LOG(LogTemp, Log, TEXT("Game ViewRect: %d x %d, Min: %d x %d"), SourceViewSize.X, SourceViewSize.Y, View.ViewRect.Min.X, View.ViewRect.Min.Y);

	if (SourceViewSize.X < DestViewSizeX || SourceViewSize.Y < DestViewSizeY)
	{
		UE_LOG(LogTemp, Log, TEXT("Source view size is smaller than destination size, skipping NRS recording."));
		return;
	}

	NRSReadbackFrameId++;

	FRDGTextureRef SceneColorTexture = (*Inputs.SceneTextures)->SceneColorTexture;
	FRDGTextureRef SceneDepthTexture = (*Inputs.SceneTextures)->SceneDepthTexture;
	FRDGTextureRef VelocityTexture = (*Inputs.SceneTextures)->GBufferVelocityTexture;

	FRDGTextureRef DestColorTexture = GraphBuilder.CreateTexture(
		FRDGTextureDesc::Create2D(
			FIntPoint(DestViewSizeX, DestViewSizeY),
			PF_R8G8B8A8,
			FClearValueBinding::Black,
			TexCreate_UAV | TexCreate_ShaderResource | TexCreate_RenderTargetable),
		TEXT("NRSRecord_ScaledSceneColor"));

	DrawDestColorTexture(GraphBuilder, InView, SceneColorTexture, DestColorTexture);

	FRDGTextureRef DestDepthTexture = GraphBuilder.CreateTexture(
		FRDGTextureDesc::Create2D(
			FIntPoint(DestViewSizeX, DestViewSizeY),
			PF_R32_FLOAT,
			FClearValueBinding::Black,
			TexCreate_UAV | TexCreate_ShaderResource | TexCreate_RenderTargetable),
		TEXT("NRSRecord_ScaledSceneDepth"));

	DrawDestDepthTexture(GraphBuilder, InView, SceneDepthTexture, DestDepthTexture);

	FRDGTextureRef DestMotionTexture = GraphBuilder.CreateTexture(
		FRDGTextureDesc::Create2D(
			FIntPoint(DestViewSizeX, DestViewSizeY),
			PF_G16R16F,
			FClearValueBinding::Black,
			TexCreate_UAV | TexCreate_ShaderResource | TexCreate_RenderTargetable),
		TEXT("NRSRecord_CameraMotion"));

	DrawDestCameraMotionTexture(GraphBuilder, InView, SceneDepthTexture, DestMotionTexture);

	if (CVarNRSRecord.GetValueOnAnyThread() != 0)
	{
		RecordBuffer(GraphBuilder, InView, DestColorTexture, SceneColorReadback, TEXT("SceneColor"));
		RecordBuffer(GraphBuilder, InView, DestDepthTexture, SceneDepthReadback, TEXT("SceneDepth"));
		RecordBuffer(GraphBuilder, InView, DestMotionTexture, CameraMotionReadback, TEXT("CameraMotion"));
	}
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

bool NRSRecordSceneViewExtension::SaveReadbackIfReady(
	NRSReadbackState& State,
	const FString& Label)
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
		State.Size.Y,
		State.Size.X,
		State.Size.Y,
		State.Size.X);
	FFileHelper::SaveArrayToFile(Output, *OutputPath);

	return true;
}

void NRSRecordSceneViewExtension::DrawDestColorTexture(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	FRDGTextureRef SourceTexture,
	FRDGTextureRef DestTexture)
{
	if (SourceTexture == nullptr || DestTexture == nullptr)
	{
		return;
	}

	const FViewInfo& View = static_cast<const FViewInfo&>(InView);

	const FScreenPassTexture OutputTexture(DestTexture);
	const FScreenPassTextureViewport OutputViewport(OutputTexture);

	// 注意, SourceTexture 的 RenderTarget 尺寸和 Viewport 尺寸不一致, 传一个 ViewRect 进去,
	// 否则会以 RenderTarget 尺寸来计算 UV, 导致采样错误
	const FScreenPassTexture InputScreenPass(SourceTexture, View.ViewRect);
	const FScreenPassTextureViewport InputViewport(InputScreenPass);

	NRSCopyColorPS::FParameters* PassParameters = GraphBuilder.AllocParameters<NRSCopyColorPS::FParameters>();
	PassParameters->InputTexture = SourceTexture;
	PassParameters->InputTextureSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->RenderTargets[0] = FRenderTargetBinding(DestTexture, ERenderTargetLoadAction::ENoAction);

	TShaderMapRef<NRSCopyColorPS> PixelShader(View.ShaderMap);

	AddDrawScreenPass(
		GraphBuilder,
		RDG_EVENT_NAME("NRS_DrawDestColor"),
		FScreenPassViewInfo(InView),
		OutputViewport,
		InputViewport,
		PixelShader,
		PassParameters);
}

void NRSRecordSceneViewExtension::DrawDestDepthTexture(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	FRDGTextureRef SourceTexture,
	FRDGTextureRef DestTexture)
{
	if (SourceTexture == nullptr || DestTexture == nullptr)
	{
		return;
	}

	const FViewInfo& View = static_cast<const FViewInfo&>(InView);

	const FScreenPassTexture OutputTexture(DestTexture);
	const FScreenPassTextureViewport OutputViewport(OutputTexture);

	// 注意, SourceTexture 的 RenderTarget 尺寸和 Viewport 尺寸不一致, 传一个 ViewRect 进去,
	// 否则会以 RenderTarget 尺寸来计算 UV, 导致采样错误
	const FScreenPassTexture InputScreenPass(SourceTexture, View.ViewRect);
	const FScreenPassTextureViewport InputViewport(InputScreenPass);

	NRSCopyColorPS::FParameters* PassParameters = GraphBuilder.AllocParameters<NRSCopyColorPS::FParameters>();
	PassParameters->InputTexture = SourceTexture;
	PassParameters->InputTextureSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->RenderTargets[0] = FRenderTargetBinding(DestTexture, ERenderTargetLoadAction::ENoAction);

	TShaderMapRef<NRSCopyColorPS> PixelShader(View.ShaderMap);

	AddDrawScreenPass(
		GraphBuilder,
		RDG_EVENT_NAME("NRS_DrawDestDepth"),
		FScreenPassViewInfo(InView),
		OutputViewport,
		InputViewport,
		PixelShader,
		PassParameters);
}

void NRSRecordSceneViewExtension::DrawDestCameraMotionTexture(
	FRDGBuilder& GraphBuilder,
	const FSceneView& InView,
	FRDGTextureRef SceneDepthTexture,
	FRDGTextureRef DestMotionTexture)
{
	if (SceneDepthTexture == nullptr || DestMotionTexture == nullptr)
	{
		return;
	}

	const FViewInfo& View = static_cast<const FViewInfo&>(InView);

	const FScreenPassTexture OutputTexture(DestMotionTexture);
	const FScreenPassTextureViewport OutputViewport(OutputTexture);
	const FScreenPassTexture InputDepth(SceneDepthTexture);
	const FScreenPassTextureViewport InputViewport(InputDepth);

	const FIntPoint DepthExtent = SceneDepthTexture->Desc.Extent;
	const float SafeSourceWidth = SourceViewSize.X > 0 ? static_cast<float>(SourceViewSize.X) : 1.0f;
	const float SafeSourceHeight = SourceViewSize.Y > 0 ? static_cast<float>(SourceViewSize.Y) : 1.0f;
	const float SafeDestWidth = DestViewSizeX > 0 ? static_cast<float>(DestViewSizeX) : 1.0f;
	const float SafeDestHeight = DestViewSizeY > 0 ? static_cast<float>(DestViewSizeY) : 1.0f;

	NRSCameraMotionPS::FParameters* PassParameters = GraphBuilder.AllocParameters<NRSCameraMotionPS::FParameters>();
	PassParameters->InputDepthTexture = SceneDepthTexture;
	PassParameters->InputDepthSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->InputViewMin = FVector2f(static_cast<float>(View.ViewRect.Min.X), static_cast<float>(View.ViewRect.Min.Y));
	PassParameters->SourceViewSize = FVector2f(static_cast<float>(SourceViewSize.X), static_cast<float>(SourceViewSize.Y));
	PassParameters->InvSourceViewSize = FVector2f(1.0f / SafeSourceWidth, 1.0f / SafeSourceHeight);
	PassParameters->DestViewSize = FVector2f(static_cast<float>(DestViewSizeX), static_cast<float>(DestViewSizeY));
	PassParameters->SourceToDestScale = FVector2f(SafeDestWidth / SafeSourceWidth, SafeDestHeight / SafeSourceHeight);
	PassParameters->InputTextureSize = FVector2f(static_cast<float>(DepthExtent.X), static_cast<float>(DepthExtent.Y));
	PassParameters->RenderTargets[0] = FRenderTargetBinding(DestMotionTexture, ERenderTargetLoadAction::ENoAction);

	TShaderMapRef<NRSCameraMotionPS> PixelShader(View.ShaderMap);

	AddDrawScreenPass(
		GraphBuilder,
		RDG_EVENT_NAME("NRS_DrawDestCameraMotion"),
		FScreenPassViewInfo(InView),
		OutputViewport,
		InputViewport,
		PixelShader,
		PassParameters);
}