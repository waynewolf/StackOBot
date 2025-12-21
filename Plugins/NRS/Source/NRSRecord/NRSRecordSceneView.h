#pragma once

#include "SceneViewExtension.h"

class FRDGBuilder;
class FSceneView;
struct FPostProcessingInputs;

class FNRSRecordSceneViewExtension : public FSceneViewExtensionBase
{
public:
	explicit FNRSRecordSceneViewExtension(const FAutoRegister& AutoRegister);

	virtual void PrePostProcessPass_RenderThread(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		const FPostProcessingInputs& Inputs) override;

	virtual void SubscribeToPostProcessingPass(
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

	void AddTranslucencyVisualization(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef TranslucencyTexture,
		FRDGTextureRef OutputTexture);

private:
	TRefCountPtr<IPooledRenderTarget> MotionVectorRT;
};
