#ifndef PROJECT_VIEWER_SETTINGS_MANAGER_H
#define PROJECT_VIEWER_SETTINGS_MANAGER_H
// System headers
//
#include <array>
#include <istream>
#include <ostream>


enum class ViewerOption
{
    ShowFrameRateInfo,
    ShowInfoPane,
    ShowLogDock,
    ShowDeveloperOptions,

    // Insert new settings here

    MAX
};

std::istream &operator>>(std::istream &s, ViewerOption &val);
std::ostream &operator<<(std::ostream &s, const ViewerOption &val);

struct ViewerOptions
{
    ViewerOptions();

    std::array<bool, static_cast<size_t>(ViewerOption::MAX)> Options;
};

std::istream &operator>>(std::istream &s, ViewerOptions &val);
std::ostream &operator<<(std::ostream &s, const ViewerOptions &val);

// Singleton that holds viewer settings
//
class ViewerSettingsManager
{
public:
    static ViewerSettingsManager &Instance()
    {
        static ViewerSettingsManager instance;
        return instance;
    }

    void SetDefaults();

    bool GetViewerOption(ViewerOption option) const
    {
        if (option == ViewerOption::MAX)
        {
            throw std::logic_error("Invalid viewer option!");
        }

        return m_settingsPayload.Options.Options[static_cast<size_t>(option)];
    }

private:
    ViewerSettingsManager(){};
    void SaveSettings(){};
    void LoadSettings(){};

    struct SettingsPayload
    {
        ViewerOptions Options;        
    };

    std::string m_settingsFilePath;
    SettingsPayload m_settingsPayload;
};



#endif
