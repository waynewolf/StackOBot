using UnrealBuildTool;
using System.IO;

public class NRSRecord : ModuleRules
{
	public NRSRecord(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"NRSCommon"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"RenderCore",
				"Renderer",
				"Projects",
				"RHI"
			}
		);

		PrivateIncludePaths.AddRange(
			new string[]
			{
				Path.Combine(EngineDirectory, "Source/Runtime/Renderer/Internal"),
				Path.Combine(EngineDirectory, "Source/Runtime/Renderer/Private"),
			}
		);
	}
}
