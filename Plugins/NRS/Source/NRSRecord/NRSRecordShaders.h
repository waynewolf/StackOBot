#pragma once

#include "GlobalShader.h"
#include "Math/IntPoint.h"
#include "Math/Vector2D.h"
#include "Math/Vector4.h"
#include "ShaderParameterStruct.h"


class NRSCopyColorPS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(NRSCopyColorPS);
	SHADER_USE_PARAMETER_STRUCT(NRSCopyColorPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, InputTextureSampler)
		SHADER_PARAMETER(FVector2f, InputViewMin)
		SHADER_PARAMETER(FVector2f, SourceViewSize)
		SHADER_PARAMETER(FVector2f, DestViewSize)
		SHADER_PARAMETER(FVector2f, InvInputTextureSize)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

class NRSCameraMotionPS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(NRSCameraMotionPS);
	SHADER_USE_PARAMETER_STRUCT(NRSCameraMotionPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputDepthTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, InputDepthSampler)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER(FVector2f, InputViewMin)
		SHADER_PARAMETER(FVector2f, SourceViewSize)
		SHADER_PARAMETER(FVector2f, InvSourceViewSize)
		SHADER_PARAMETER(FVector2f, DestViewSize)
		SHADER_PARAMETER(FVector2f, SourceToDestScale)
		SHADER_PARAMETER(FVector2f, InputTextureSize)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

