#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "Misc/CoreDelegates.h"
#include "NRSRecordSceneView.h"

class NRSRecordSceneViewExtension;
class NRSRecordModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	void HandlePostEngineInit();

private:
	TSharedPtr<NRSRecordSceneViewExtension, ESPMode::ThreadSafe> SceneView;
	FDelegateHandle PostEngineInitHandle;
};
