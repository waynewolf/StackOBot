#pragma once

#include "SceneViewExtension.h"

struct NRSReadbackState
{
	TUniquePtr<FRHIGPUTextureReadback> Readback;
	FIntPoint Size = FIntPoint::ZeroValue;
	EPixelFormat Format = PF_Unknown;
	uint64 FrameId = 0;
	bool bPendingCopy = false;
	bool bNeedsReset = false;
	FIntPoint PendingSize = FIntPoint::ZeroValue;
	EPixelFormat PendingFormat = PF_Unknown;

	explicit NRSReadbackState(const TCHAR* Name)
		: Readback(MakeUnique<FRHIGPUTextureReadback>(Name))
		, Name(Name)
	{
	}

	const TCHAR* Name = nullptr;
};

class FRDGBuilder;
class FSceneView;
struct FPostProcessMaterialInputs;
struct FPostProcessingInputs;
class NRSRecordSceneViewExtension : public FSceneViewExtensionBase
{
public:
	explicit NRSRecordSceneViewExtension(const FAutoRegister& AutoRegister);

	void PreRenderView_RenderThread(FRDGBuilder& GraphBuilder, FSceneView& InView) override;

	void PrePostProcessPass_RenderThread(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		const FPostProcessingInputs& Inputs) override;

	void RecordBuffer(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef InTexture,
		NRSReadbackState& OutReadbackState,
		const FString& Label);

	bool SaveReadbackIfReady(
		NRSReadbackState& State,
		const FString& Label);

	void DrawDestColorTexture(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SourceTexture,
		FRDGTextureRef DestTexture);

	void DrawDestDepthTexture(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SourceTexture,
		FRDGTextureRef DestTexture);

	void DrawDestCameraMotionTexture(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SceneDepthTexture,
		FRDGTextureRef VelocityTexture,
		FRDGTextureRef DestMotionTexture);

private:
	static uint64 NRSReadbackFrameId;

	NRSReadbackState SceneColorReadback;
	NRSReadbackState SceneDepthReadback;
	NRSReadbackState CameraMotionReadback;

	static int DestViewSizeX;
	static int DestViewSizeY;
};
