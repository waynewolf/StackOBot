#include "NRSRecordSceneView.h"
#include "NRSRecordShaders.h"

#include "PostProcess/PostProcessInputs.h"
#include "PostProcess/PostProcessMaterialInputs.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "ScreenPass.h"
#include "SceneView.h"
#include "SceneRendering.h"

IMPLEMENT_GLOBAL_SHADER(FNRSMotionGenCS, "/Plugin/NRS/MotionGen.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FNRSMotionVizPS, "/Plugin/NRS/MotionViz.usf", "MainPS", SF_Pixel);
IMPLEMENT_GLOBAL_SHADER(FNRSVisualizeXPS, "/Plugin/NRS/VisualizeX.usf", "MainPS", SF_Pixel);

FNRSRecordSceneViewExtension::FNRSRecordSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister)
{
}

void FNRSRecordSceneViewExtension::PrePostProcessPass_RenderThread(
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
	}

	GraphBuilder.QueueTextureExtraction(MotionVectorTexture, &MotionVectorRT);
}

void FNRSRecordSceneViewExtension::SubscribeToPostProcessingPass(
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
		InOutPassCallbacks.Add(FPostProcessingPassDelegate::CreateRaw(this, &FNRSRecordSceneViewExtension::InPostProcessChain));
	}
}

FScreenPassTexture FNRSRecordSceneViewExtension::InPostProcessChain(
	FRDGBuilder& GraphBuilder,
	const FSceneView& View,
	const FPostProcessMaterialInputs& Inputs)
{
	RDG_EVENT_SCOPE(GraphBuilder, "InPostProcessChain");

	FScreenPassTexture SceneColorScreenPassTexture(Inputs.GetInput(EPostProcessMaterialInput::SceneColor));
	FScreenPassTexture TranslucencyScreenPassTexture(Inputs.GetInput(EPostProcessMaterialInput::SeparateTranslucency));

	AddTranslucencyVisualization(GraphBuilder, View, TranslucencyScreenPassTexture.Texture, SceneColorScreenPassTexture.Texture);

	return SceneColorScreenPassTexture;
}

void FNRSRecordSceneViewExtension::AddMotionGeneration(
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

	FNRSMotionGenCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNRSMotionGenCS::FParameters>();
	PassParameters->DepthTexture = SceneDepthTexture;
	PassParameters->InputDepth = SceneDepthSRV;
	PassParameters->InputVelocity = SceneVelocitySRV;
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->OutputTexture = MotionVectorUAV;

	TShaderMapRef<FNRSMotionGenCS> ComputeShader(View.ShaderMap);
	FComputeShaderUtils::AddPass(
		GraphBuilder,
		RDG_EVENT_NAME("MotionGen"),
		ComputeShader,
		PassParameters,
		FComputeShaderUtils::GetGroupCount(
			FIntVector(ViewSize.X, ViewSize.Y, 1),
			FIntVector(FNRSMotionGenCS::ThreadgroupSizeX, FNRSMotionGenCS::ThreadgroupSizeY, FNRSMotionGenCS::ThreadgroupSizeZ))
	);
}

void FNRSRecordSceneViewExtension::AddMotionVisualization(
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

	FNRSMotionVizPS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNRSMotionVizPS::FParameters>();
	PassParameters->InputMotion = MotionVectorTexture;
	PassParameters->InputMotionSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->InputColor = SceneColorCopy;
	PassParameters->InputColorSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->RenderTargets[0] = Output.GetRenderTargetBinding();

	TShaderMapRef<FNRSMotionVizPS> PixelShader(View.ShaderMap);
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

void FNRSRecordSceneViewExtension::AddTranslucencyVisualization(
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

	FNRSVisualizeXPS::FParameters* PassParameters = GraphBuilder.AllocParameters<FNRSVisualizeXPS::FParameters>();
	PassParameters->InputTexture = TranslucencyScreenPassTexture.Texture;
	PassParameters->InputTextureSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
	PassParameters->RenderTargets[0] = OutputRT.GetRenderTargetBinding();

	TShaderMapRef<FNRSVisualizeXPS> PixelShader(View.ShaderMap);
	AddDrawScreenPass(
		GraphBuilder,
		RDG_EVENT_NAME("TranslucencyViz"),
		FScreenPassViewInfo(InView),
		OutputViewport,
		OutputViewport,
		PixelShader,
		PassParameters
	);
}
