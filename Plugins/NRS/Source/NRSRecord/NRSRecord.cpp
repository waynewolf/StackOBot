#include "NRSRecord.h"
#include "Interfaces/IPluginManager.h"
#include "Modules/ModuleManager.h"
#include "ShaderCore.h"

IMPLEMENT_MODULE(NRSRecordModule, NRSRecord)

void NRSRecordModule::StartupModule()
{
	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("NRS"));
	if (Plugin.IsValid())
	{
		const FString ShaderDir = FPaths::Combine(Plugin->GetBaseDir(), TEXT("Shaders"));
		AddShaderSourceDirectoryMapping(TEXT("/Plugin/NRS"), ShaderDir);
	}

	if (GEngine)
	{
		SceneView = FSceneViewExtensions::NewExtension<NRSRecordSceneViewExtension>();
	}
	else
	{
		PostEngineInitHandle = FCoreDelegates::OnPostEngineInit.AddRaw(this, &NRSRecordModule::HandlePostEngineInit);
	}

	UE_LOG(LogTemp, Log, TEXT("NRSRecord module has started."));
}

void NRSRecordModule::ShutdownModule()
{
	UE_LOG(LogTemp, Log, TEXT("NRSRecord module is shutting down."));

	if (PostEngineInitHandle.IsValid())
	{
		FCoreDelegates::OnPostEngineInit.Remove(PostEngineInitHandle);
		PostEngineInitHandle.Reset();
	}

	SceneView.Reset();
}

void NRSRecordModule::HandlePostEngineInit()
{
	UE_LOG(LogTemp, Warning, TEXT("HandlePostEngineInit"));

	PostEngineInitHandle.Reset();

	SceneView = FSceneViewExtensions::NewExtension<NRSRecordSceneViewExtension>();
}
