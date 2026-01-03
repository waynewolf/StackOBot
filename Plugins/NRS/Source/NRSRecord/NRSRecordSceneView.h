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

	void PrePostProcessPass_RenderThread(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		const FPostProcessingInputs& Inputs) override;

private:
	void AddMotionGeneration(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SceneDepthTexture,
		FRDGTextureRef SceneVelocityTexture,
		FRDGTextureRef MotionVectorTexture);

	void RecordDepthBuffer(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef InTexture,
		NRSReadbackState& OutReadbackState,
		const FString& Label);

	void RecordBuffer(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef InTexture,
		NRSReadbackState& OutReadbackState,
		const FString& Label);

	bool SaveReadbackIfReady(NRSReadbackState& State, const FString& Label);

private:
	static uint64 NRSReadbackFrameId;

	TRefCountPtr<IPooledRenderTarget> MotionVectorRT;

	NRSReadbackState SceneColorReadback;
	NRSReadbackState SceneDepthReadback;
	NRSReadbackState CameraMotionReadback;

	FIntPoint ViewSize;
};
