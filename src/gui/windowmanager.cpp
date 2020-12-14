//
// Created by ynki9 on 12/7/20.
//

#include "gui/windowmanager.h"
#include "gui/viewersettingsmanager.h"
#include "gui/windowsizehelper.h"

WindowManager &WindowManager::Instance(){
    static WindowManager instance;
    return instance;
}

void WindowManager::SetGLWindowSize(ImVec2 glWindowSize)
{
    glWindowSize_ = glWindowSize;
}

void WindowManager::SetMenuBarHeight(float menuBarHeight)
{
    menuBarHeight = menuBarHeight;
}

void WindowManager::AddWindow(std::unique_ptr<VisualizationWindow> &&window)
{
    windows_.WindowGroup.emplace_back(WindowListEntry(std::move(window)));
}

void WindowManager::AddWindowGroup(std::vector<std::unique_ptr<VisualizationWindow>> &&windowGroup)
{
    windows_.WindowGroup.emplace_back(std::move(windowGroup));
}

void WindowManager::ClearFullscreenWindow()
{
    maximizedWindow_ = nullptr;
}

void WindowManager::ClearWindows()
{
    assert(windows_.IsWindowGroup);
    windows_.WindowGroup.clear();
    ClearFullscreenWindow();
}

void WindowManager::PushLeftDockControl(std::unique_ptr<DockControl> &&dockControl)
{
    leftDock_.PushDockControl(std::move(dockControl));
}

void WindowManager::PushBottomDockControl(std::unique_ptr<DockControl> &&dockControl)
{
    bottomDock_.PushDockControl(std::move(dockControl));
}


void WindowManager::ShowAll()
{
    const ImVec2 leftDockRegionPos(0.f, menuBarHeight_);
    const ImVec2 leftDockRegionSize(glWindowSize_.x, glWindowSize_.y - leftDockRegionPos.y);
    leftDock_.Show(leftDockRegionPos, leftDockRegionSize);

    const ImVec2 windowAreaPosition(leftDock_.GetSize().x, menuBarHeight_);
    ImVec2 windowAreaSize(glWindowSize_.x - windowAreaPosition.x, glWindowSize_.y - windowAreaPosition.y);

    if (ViewerSettingsManager::Instance().GetViewerOption(ViewerOption::ShowLogDock))
    {
        const ImVec2 bottomDockRegionPos(leftDock_.GetSize().x, menuBarHeight_);
        const ImVec2 bottomDockRegionSize(glWindowSize_.x - bottomDockRegionPos.x,
                                          glWindowSize_.y - bottomDockRegionPos.y);
        bottomDock_.Show(bottomDockRegionPos, bottomDockRegionSize);

        windowAreaSize.y -= bottomDock_.GetSize().y;
    }

    if (maximizedWindow_ != nullptr)
    {
        ShowWindow(windowAreaPosition, windowAreaSize, maximizedWindow_, true);
    }
    else
    {
        ShowWindowArea(windowAreaPosition, windowAreaSize, &windows_);
    }
}



void WindowManager::ShowWindowArea(ImVec2 windowAreaPosition, ImVec2 windowAreaSize, WindowListEntry *windowList)
{
    if (!windowList->IsWindowGroup)
    {
        ShowWindow(windowAreaPosition, windowAreaSize, windowList->Window.get(), false);
        return;
    }

    ImVec2 individualWindowSize = windowAreaSize;

    int totalRows = 1;
    int totalColumns = 1;

    bool nextDivisionHorizontal = false;

    size_t divisionsRemaining = windowList->WindowGroup.size();
    while (divisionsRemaining > 1)
    {
        if (nextDivisionHorizontal)
        {
            totalRows++;
        }
        else
        {
            totalColumns++;
        }

        divisionsRemaining = (divisionsRemaining / 2) + (divisionsRemaining % 2);
        nextDivisionHorizontal = !nextDivisionHorizontal;
    }

    individualWindowSize.x /= totalColumns;
    individualWindowSize.y /= totalRows;

    int currentRow = 0;
    int currentColumn = 0;
    for (auto &listEntry : windowList->WindowGroup)
    {
        ImVec2 entryPosition = { windowAreaPosition.x + currentColumn * individualWindowSize.x,
                                 windowAreaPosition.y + currentRow * individualWindowSize.y };

        ImVec2 entrySize = individualWindowSize;

        currentColumn = (currentColumn + 1) % totalColumns;
        currentRow = currentRow + (currentColumn == 0 ? 1 : 0);

        if (listEntry.IsWindowGroup)
        {
            ShowWindowArea(entryPosition, entrySize, &listEntry);
        }
        else
        {
            ShowWindow(entryPosition, entrySize, listEntry.Window.get(), false);
        }
    }
}


void WindowManager::ShowWindow(const ImVec2 windowAreaPosition,
                                  const ImVec2 windowAreaSize,
                                  VisualizationWindow *window,
                                  bool isMaximized)
{
    WindowPlacementInfo placementInfo;
    placementInfo.Position = windowAreaPosition;
    placementInfo.Size = windowAreaSize;
    placementInfo.Size.y -= GetTitleBarHeight();
    placementInfo.Size.x = std::max(1.f, placementInfo.Size.x);
    placementInfo.Size.y = std::max(1.f, placementInfo.Size.y);
    ImGui::SetNextWindowPos(windowAreaPosition);
    ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0), windowAreaSize);

    static constexpr ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
                                                    ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::PushStyleColor(ImGuiCol_TitleBg, ImGui::GetStyleColorVec4(ImGuiCol_TitleBgActive));

    if (ImGui::Begin(window->GetTitle(), nullptr, windowFlags))
    {
        window->Show(placementInfo);

        // Draw minimize/maximize button
        //
        if (windows_.WindowGroup.size() != 1)
        {
            if (ShowMinMaxButton("-", "+", isMaximized))
            {
                if (maximizedWindow_
                == nullptr)
                {
                    maximizedWindow_ = window;
                }
                else
                {
                    ClearFullscreenWindow();
                }
            }
        }
    }
    ImGui::End();

    ImGui::PopStyleColor();
}

bool WindowManager::ShowMinMaxButton(const char *minimizeLabel, const char *maximizeLabel, bool isMaximized)
{
    bool result = false;

    const char *label = isMaximized ? minimizeLabel : maximizeLabel;

    const char *parentWindowTitle = ImGui::GetCurrentWindow()->Name;
    const ImVec2 currentWindowSize = ImGui::GetWindowSize();
    const ImVec2 currentWindowPosition = ImGui::GetWindowPos();
    // Make the button fit inside the border of the parent window
    //
    const float windowBorderSize = ImGui::GetStyle().WindowBorderSize;
    const float buttonSize = GetTitleBarHeight() - (2 * windowBorderSize);
    ImVec2 minMaxButtonSize(buttonSize, buttonSize);
    minMaxButtonSize.x = std::min(minMaxButtonSize.x, currentWindowSize.x);
    minMaxButtonSize.y = std::min(minMaxButtonSize.y, currentWindowSize.y);

    const ImVec2 minMaxPosition(currentWindowPosition.x + currentWindowSize.x - minMaxButtonSize.x - windowBorderSize,
                                currentWindowPosition.y + windowBorderSize);

    constexpr ImGuiWindowFlags minMaxButtonFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                                                   ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
                                                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings;

    const std::string minMaxButtonTitle = std::string(parentWindowTitle) + "##minmax";

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

    ImGui::SetNextWindowPos(minMaxPosition, ImGuiCond_Always);
    ImGui::SetNextWindowSize(minMaxButtonSize);
    if (ImGui::Begin(minMaxButtonTitle.c_str(), nullptr, minMaxButtonFlags))
    {
        if (ImGui::Button(label, minMaxButtonSize))
        {
            result = true;
        }
    }
    ImGui::End();

    ImGui::PopStyleVar(2);

    return result;
}
