#include "NRSRecordSceneView.h"
#include "NRSRecordShaders.h"

#include "PostProcess/PostProcessInputs.h"
#include "RenderGraphBuilder.h"
#include "SceneView.h"
#include "SceneRendering.h"

IMPLEMENT_GLOBAL_SHADER(FNRSMotionGenCS, "/Plugin/NRS/MotionGen.usf", "MainCS", SF_Compute);

FNRSRecordSceneViewExtension::FNRSRecordSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister)
{
}

void FNRSRecordSceneViewExtension::PrePostProcessPass_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& InView, const FPostProcessingInputs& Inputs)
{
	const FViewInfo& View = (FViewInfo &)InView;

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

	FNRSMotionGenCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNRSMotionGenCS::FParameters>();
	PassParameters->DepthTexture = SceneDepthTexture;
	PassParameters->InputDepth = GraphBuilder.CreateSRV(SceneDepthTexture);
	PassParameters->InputVelocity = GraphBuilder.CreateSRV(VelocityTexture);
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->OutputTexture = GraphBuilder.CreateUAV(MotionVectorTexture);

	TShaderMapRef<FNRSMotionGenCS> ComputeShader(View.ShaderMap);
	FComputeShaderUtils::AddPass(
		GraphBuilder,
		RDG_EVENT_NAME("MotionGen"),
		ComputeShader,
		PassParameters,
		FComputeShaderUtils::GetGroupCount(
			FIntVector(SceneDepthTexture->Desc.Extent.X, SceneDepthTexture->Desc.Extent.Y, 1),
			FIntVector(FNRSMotionGenCS::ThreadgroupSizeX, FNRSMotionGenCS::ThreadgroupSizeY, FNRSMotionGenCS::ThreadgroupSizeZ))
	);
	
	GraphBuilder.QueueTextureExtraction(MotionVectorTexture, &MotionVectorRT);
}
