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

	void SubscribeToPostProcessingPass(
		EPostProcessingPass Pass,
		const FSceneView& InView,
		FPostProcessingPassDelegateArray& InOutPassCallbacks,
		bool bIsPassEnabled) override;

private:
	FScreenPassTexture InPostProcessChain(
		FRDGBuilder& GraphBuilder,
		const FSceneView& View,
		const FPostProcessMaterialInputs& Inputs);

	void AddMotionGeneration(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SceneDepthTexture,
		FRDGTextureRef SceneVelocityTexture,
		FRDGTextureRef MotionVectorTexture);
	
	void AddMotionVisualization(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SceneColorTexture,
		FRDGTextureRef SceneDepthTexture,
		FRDGTextureRef MotionVectorTexture);

	void AddXVisualization(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef TranslucencyTexture,
		FRDGTextureRef OutputTexture);

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
	NRSReadbackState MotionVectorReadback;
	NRSReadbackState TranslucencyReadback;
	NRSReadbackState GBufferCReadback;

	FIntPoint ViewSize;
};
