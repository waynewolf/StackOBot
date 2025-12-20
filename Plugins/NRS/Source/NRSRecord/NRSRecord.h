#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "Misc/CoreDelegates.h"
#include "NRSRecordSceneView.h"

class FNRSRecordSceneViewExtension;
class FNRSRecordModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	void HandlePostEngineInit();

private:
	TSharedPtr<FNRSRecordSceneViewExtension, ESPMode::ThreadSafe> SceneView;
	FDelegateHandle PostEngineInitHandle;
};
