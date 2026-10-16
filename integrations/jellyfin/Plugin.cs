using MediaBrowser.Common.Configuration;
using MediaBrowser.Common.Plugins;
using MediaBrowser.Model.Plugins;
using MediaBrowser.Model.Serialization;

namespace Jellyfin.Plugin.DubAnything;

public sealed class PluginConfiguration : BasePluginConfiguration
{
    public string ServiceUrl { get; set; } = "http://127.0.0.1:8767";
    public string ServiceToken { get; set; } = "";
    public string DefaultDirector { get; set; } = "codex";
    public string CodexModel { get; set; } = "";
    public string ClaudeModel { get; set; } = "";
    public bool ReviewBeforePublish { get; set; } = true;
    public string[] RefreshedJobIds { get; set; } = [];
}

public sealed class Plugin : BasePlugin<PluginConfiguration>, IHasWebPages
{
    public Plugin(IApplicationPaths paths, IXmlSerializer serializer) : base(paths, serializer)
    {
        Instance = this;
    }

    public static Plugin? Instance { get; private set; }
    public override string Name => "Dub Anything";
    public override string Description => "Dub library videos with a headless Codex or Claude Code director.";
    public override Guid Id => Guid.Parse("6d52a0cd-425d-4802-ad0c-368a83da8e45");

    public IEnumerable<PluginPageInfo> GetPages() =>
    [
        new PluginPageInfo
        {
            Name = "dubanything",
            DisplayName = "Dub Anything",
            EmbeddedResourcePath = GetType().Namespace + ".Configuration.configPage.html",
            EnableInMainMenu = true,
            MenuSection = "server",
            MenuIcon = "record_voice_over"
        }
    ];
}
