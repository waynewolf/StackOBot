#pragma once

#include "Modules/ModuleManager.h"

class NRSSuperFrameModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
