// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Associated header
//
#include "imguiextensions.h"

// System headers
//
#include <functional>
#include <sstream>

// Library headers
//
#include "imgui_all.h"

// Project headers
//
//#include "filesystem17.h"

namespace
{
std::string ConvertToVerticalText(const char *str)
{
    std::stringstream ss;
    bool first = true;
    while (*str)
    {
        if (first)
        {
            first = false;
        }
        else
        {
            ss << "\n";
        }

        ss << *str;
        ++str;
    }
    return ss.str();
}
} // namespace

namespace viewer
{
namespace ImGuiExtensions
{

bool Button(const char *label, const bool enabled)
{
    return Button(label, ImVec2(0, 0), enabled);
}

bool Button(const char *label, const ImVec2 &size, bool enabled)
{
    return ShowDisableableControl<bool>([label, &size]() { return ImGui::Button(label, size); }, enabled);
}

bool Checkbox(const char *label, bool *checked, const bool enabled)
{
    return ShowDisableableControl<bool>([label, checked]() { return ImGui::Checkbox(label, checked); }, enabled);
}

bool RadioButton(const char *label, bool active, bool enabled)
{
    return ShowDisableableControl<bool>([label, active]() { return ImGui::RadioButton(label, active); }, enabled);
}

bool RadioButton(const char *label, int *v, int vButton, bool enabled)
{
    return ShowDisableableControl<bool>([label, v, vButton]() { return ImGui::RadioButton(label, v, vButton); },
                                        enabled);
}

bool InputScalar(const char *label,
                    ImGuiDataType dataType,
                    void *dataPtr,
                    const void *step,
                    const void *stepFast,
                    const char *format,
                    bool enabled)
{
    return ShowDisableableControl<bool>(
        [&]() { return ImGui::InputScalar(label, dataType, dataPtr, step, stepFast, format); }, enabled);
}

bool SliderInt(const char *label, int *value, int valueMin, int valueMax, const char *format, bool enabled)
{
    return ShowDisableableControl<bool>([&]() { return ImGui::SliderInt(label, value, valueMin, valueMax, format); },
                                        enabled);
}

bool SliderFloat(const char *label,
                    float *value,
                    float valueMin,
                    float valueMax,
                    const char *format,
                    float power,
                    bool enabled)
{
    return ShowDisableableControl<bool>(
        [&]() { return ImGui::SliderFloat(label, value, valueMin, valueMax, format, power); }, enabled);
}

bool VSliderFloat(const char *name, ImVec2 size, float *value, float minValue, float maxValue, const char *label)
{
    const std::string vLabel = ConvertToVerticalText(label);
    return ImGui::VSliderFloat(name, size, value, minValue, maxValue, vLabel.c_str());
}

void VText(const char *s)
{
    const std::string vLabel = ConvertToVerticalText(s);
    ImGui::Text("%s", vLabel.c_str());
}

void ShowTooltip(const char *msg, bool show)
{
    if (show && ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(msg);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

} // namespace ImGuiExtensions

