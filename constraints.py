from __future__ import annotations
from collections import defaultdict

def penalty_morning_preference(schedule) -> float:
    return sum(1 for c in schedule.classes if c.instructor.prefers_morning and not c.meeting_time.is_morning)

def penalty_late_classes(schedule) -> float:
    return sum(1 for c in schedule.classes if c.meeting_time.is_late)

def penalty_instructor_overload(schedule) -> float:
    instructor_day_counts = defaultdict(int)
    for c in schedule.classes:
        instructor_day_counts[f"{c.instructor.id}_{c.meeting_time.day}"] += 1
    return sum((count - 3) * 2 for count in instructor_day_counts.values() if count > 3)

def penalty_smart_breaks(schedule) -> float:
    lunch_hour = schedule.data.lunch_hour
    penalties = 0.0
    section_day = defaultdict(list)
    for c in schedule.classes: section_day[f"{c.section.id}_{c.meeting_time.day}"].append(c)

    for classes in section_day.values():
        if len(classes) < 2: continue
        classes.sort(key=lambda x: x.meeting_time.hour)
        consecutive = 1
        for i in range(1, len(classes)):
            cur_hr, prv_hr = classes[i].meeting_time.hour, classes[i - 1].meeting_time.hour
            consecutive = consecutive + 1 if cur_hr == prv_hr + 1 else 1
            limit = 2 if cur_hr < lunch_hour else 3
            if consecutive > limit: penalties += 1.5
    return penalties

def penalty_instructor_gaps(schedule) -> float:
    penalties = 0.0
    instructor_day = defaultdict(list)
    for c in schedule.classes: instructor_day[f"{c.instructor.id}_{c.meeting_time.day}"].append(c.meeting_time.hour)
    for hours in instructor_day.values():
        if len(hours) < 2: continue
        hours.sort()
        for i in range(1, len(hours)):
            gap = hours[i] - hours[i - 1]
            if gap >= 3: penalties += gap - 2
    return penalties

def penalty_section_gaps(schedule) -> float:
    penalties = 0.0
    section_day = defaultdict(list)
    for c in schedule.classes: section_day[f"{c.section.id}_{c.meeting_time.day}"].append(c.meeting_time.hour)
    for hours_raw in section_day.values():
        hours = sorted(set(hours_raw))
        for i in range(1, len(hours)):
            gap = hours[i] - hours[i - 1]
            if gap >= 3: penalties += (gap - 2) * 1.5
    return penalties

def penalty_minor_course_spread(schedule) -> float:
    penalties = 0.0
    section_day_load = defaultdict(int)
    for c in schedule.classes:
        if c.course.course_type != "Minor":
            section_day_load[f"{c.section.id}_{c.meeting_time.day}"] += 1
    for c in schedule.classes:
        if c.course.course_type != "Minor": continue
        day = c.meeting_time.day
        for sec_id in c.get_affected_section_ids():
            load = section_day_load.get(f"{sec_id}_{day}", 0)
            if load >= 5: penalties += (load - 4) * 0.5
    return penalties

def penalty_lab_lecture_same_day(schedule) -> float:
    penalties = 0.0
    lecture_day = defaultdict(set)
    for c in schedule.classes:
        if c.course.course_type in ("Lecture", "Minor") and c.batch == "ALL":
            lecture_day[f"{c.section.id}_{c.meeting_time.day}"].add(c.course.id)
    for c in schedule.classes:
        if c.course.course_type == "Lab":
            if c.course.id in lecture_day.get(f"{c.section.id}_{c.meeting_time.day}", set()):
                penalties += 1.0
    return penalties

SOFT_CONSTRAINTS_REGISTRY: dict[str, callable] = {
    "Respect Morning Preferences":                          penalty_morning_preference,
    "Minimize Late Classes (after 3 PM)":                   penalty_late_classes,
    "Avoid Instructor Daily Overload (>3 classes)":         penalty_instructor_overload,
    "Enforce Smart Breaks (max 2 AM / 3 PM streak)":        penalty_smart_breaks,
    "Reduce Instructor Mid-Day Gaps":                       penalty_instructor_gaps,
    "Reduce Student Idle Gaps":                             penalty_section_gaps,
    "Avoid Heavy Days for Minor Course Sections":           penalty_minor_course_spread,
    "Avoid Lab and Lecture Same Day (rotation model)":      penalty_lab_lecture_same_day,
}