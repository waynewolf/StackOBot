#pragma once

#include "SceneViewExtension.h"
#include "PostProcess/PostProcessInputs.h"

class FRDGBuilder;
class FSceneView;

class NRSRecordSceneViewExtension : public FSceneViewExtensionBase
{
public:
	explicit NRSRecordSceneViewExtension(const FAutoRegister& AutoRegister);

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

	void AddXVisualization(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef TranslucencyTexture,
		FRDGTextureRef OutputTexture);
	
	void RecordBuffers(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		FRDGTextureRef SceneColorTexture,
		FRDGTextureRef SceneDepthTexture,
		FRDGTextureRef MotionVectorTexture,
		FRDGTextureRef TranslucencyTexture,
		FRDGTextureRef GBufferATexture,
		FRDGTextureRef GBufferBTexture,
		FRDGTextureRef GBufferCTexture);

private:
	FPostProcessingInputs CachedPPInputs;
	FRDGTextureRef CachedMotionVectorTexture;
	TRefCountPtr<IPooledRenderTarget> MotionVectorRT;
};
