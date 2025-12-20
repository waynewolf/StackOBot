#pragma once

#include "Modules/ModuleManager.h"

class FNRSCommonModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
