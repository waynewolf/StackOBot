#pragma once

#include "SceneViewExtension.h"

class FRDGBuilder;
class FSceneView;
struct FPostProcessingInputs;

class FNRSRecordSceneViewExtension : public FSceneViewExtensionBase
{
public:
	explicit FNRSRecordSceneViewExtension(const FAutoRegister& AutoRegister);

	virtual void PrePostProcessPass_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& InView, const FPostProcessingInputs& Inputs) override;

private:
	TRefCountPtr<IPooledRenderTarget> MotionVectorRT;
};
