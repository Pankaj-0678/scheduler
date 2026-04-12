"""
Soft constraint functions for the timetable GA.

Each function receives a Schedule and returns a non-negative float penalty.
Higher penalty = worse schedule.  Zero = fully satisfied.

Changes vs previous version:
  - penalty_lab_lecture_same_day (NEW): penalise any batch that has both
    the lecture and the lab session for the same subject on the same day.
    In a rotation model the lab and theory components of a course ideally
    fall on different days to avoid cognitive overload and to leave a day
    gap for pre-lab preparation.
  - penalty_minor_course_spread: unchanged.
  - All other constraints: unchanged.
"""

from __future__ import annotations
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Instructor morning preference
# ---------------------------------------------------------------------------

def penalty_morning_preference(schedule) -> float:
    """
    Penalise every class where an instructor who prefers mornings is
    assigned an afternoon slot.
    """
    penalties = 0
    for c in schedule.classes:
        if c.instructor.prefers_morning and not c.meeting_time.is_morning:
            penalties += 1
    return float(penalties)


# ---------------------------------------------------------------------------
# 2. Late classes (after 15:00)
# ---------------------------------------------------------------------------

def penalty_late_classes(schedule) -> float:
    """Penalise any class scheduled at or after 15:00."""
    penalties = 0
    for c in schedule.classes:
        if c.meeting_time.is_late:
            penalties += 1
    return float(penalties)


# ---------------------------------------------------------------------------
# 3. Instructor daily overload
# ---------------------------------------------------------------------------

def penalty_instructor_overload(schedule) -> float:
    """
    Penalise instructors with more than 3 classes on any single day.
    Uses a steeper slope beyond the limit.
    """
    instructor_day_counts: dict[str, int] = defaultdict(int)
    for c in schedule.classes:
        key = f"{c.instructor.id}_{c.meeting_time.day}"
        instructor_day_counts[key] += 1

    penalties = 0
    for count in instructor_day_counts.values():
        if count > 3:
            penalties += (count - 3) * 2
    return float(penalties)


# ---------------------------------------------------------------------------
# 4. Smart break enforcement
# ---------------------------------------------------------------------------

def penalty_smart_breaks(schedule) -> float:
    """
    Discourage very long consecutive class runs:
      - More than 2 consecutive classes before lunch → penalty
      - More than 3 consecutive classes after lunch  → penalty
    """
    lunch_hour = schedule.data.lunch_hour
    penalties = 0.0

    section_day: dict[str, list] = defaultdict(list)
    for c in schedule.classes:
        key = f"{c.section.id}_{c.meeting_time.day}"
        section_day[key].append(c)

    for classes in section_day.values():
        if len(classes) < 2:
            continue
        classes.sort(key=lambda x: x.meeting_time.hour)

        consecutive = 1
        for i in range(1, len(classes)):
            cur_hr = classes[i].meeting_time.hour
            prv_hr = classes[i - 1].meeting_time.hour

            if cur_hr == prv_hr + 1:
                consecutive += 1
            else:
                consecutive = 1

            limit = 2 if cur_hr < lunch_hour else 3
            if consecutive > limit:
                penalties += 1.5

    return penalties


# ---------------------------------------------------------------------------
# 5. Instructor mid-day gaps
# ---------------------------------------------------------------------------

def penalty_instructor_gaps(schedule) -> float:
    """
    Penalise large idle gaps (≥3 hours) between an instructor's classes
    on the same day.
    """
    penalties = 0.0
    instructor_day: dict[str, list] = defaultdict(list)

    for c in schedule.classes:
        key = f"{c.instructor.id}_{c.meeting_time.day}"
        instructor_day[key].append(c.meeting_time.hour)

    for hours in instructor_day.values():
        if len(hours) < 2:
            continue
        hours.sort()
        for i in range(1, len(hours)):
            gap = hours[i] - hours[i - 1]
            if gap >= 3:
                penalties += gap - 2

    return penalties


# ---------------------------------------------------------------------------
# 6. Section compactness (student idle gaps)
# ---------------------------------------------------------------------------

def penalty_section_gaps(schedule) -> float:
    """
    Students should not have a 2+ hour hole in their timetable.
    Penalise every such gap per section per day.
    """
    penalties = 0.0
    section_day: dict[str, list] = defaultdict(list)

    for c in schedule.classes:
        key = f"{c.section.id}_{c.meeting_time.day}"
        section_day[key].append(c.meeting_time.hour)

    for hours_raw in section_day.values():
        hours = sorted(set(hours_raw))
        for i in range(1, len(hours)):
            gap = hours[i] - hours[i - 1]
            if gap >= 3:
                penalties += (gap - 2) * 1.5

    return penalties


# ---------------------------------------------------------------------------
# 7. Minor course scheduling quality
# ---------------------------------------------------------------------------

def penalty_minor_course_spread(schedule) -> float:
    """
    Minor course electives should not be crammed into already-heavy days
    for any of their enrolled sections.

    Penalty is proportional to how many events each affected section
    already has on the minor course's scheduled day.
    """
    penalties = 0.0

    # Build per-section, per-day event count (non-minor)
    section_day_load: dict[str, int] = defaultdict(int)
    for c in schedule.classes:
        if c.course.course_type != "Minor":
            key = f"{c.section.id}_{c.meeting_time.day}"
            section_day_load[key] += 1

    for c in schedule.classes:
        if c.course.course_type != "Minor":
            continue

        day = c.meeting_time.day
        affected_ids = c.get_affected_section_ids()
        for sec_id in affected_ids:
            load = section_day_load.get(f"{sec_id}_{day}", 0)
            if load >= 5:
                penalties += (load - 4) * 0.5

    return penalties


# ---------------------------------------------------------------------------
# 8. Lab and lecture for the same subject on the same day  (NEW)
# ---------------------------------------------------------------------------

def penalty_lab_lecture_same_day(schedule) -> float:
    """
    In a rotation timetable each batch should ideally have its lab session
    for subject X on a DIFFERENT day from the shared lecture for subject X.

    Reason: combining the 2-hour lab and 1-hour lecture back-to-back on the
    same day overloads students cognitively, leaves no time for pre-lab
    preparation, and compresses the rest of the week unevenly.

    How it works:
      1. Build a lookup of (section_id, day) → {course_ids with a lecture}.
      2. For every lab event, check whether the same course has a lecture
         on the same day for that section.  Each match adds a penalty of 1.

    Note: this is a *soft* constraint.  Some institutions deliberately
    schedule the lecture immediately before the lab (e.g. Mon lecture then
    Mon afternoon lab).  Adjust the weight in the registry to taste —
    set it to 0 to disable entirely.
    """
    penalties = 0.0

    # (section_id, day) → set of course ids that have a lecture that day
    lecture_day: dict[str, set] = defaultdict(set)
    for c in schedule.classes:
        if c.course.course_type in ("Lecture", "Minor") and c.batch == "ALL":
            key = f"{c.section.id}_{c.meeting_time.day}"
            lecture_day[key].add(c.course.id)

    for c in schedule.classes:
        if c.course.course_type != "Lab":
            continue
        key = f"{c.section.id}_{c.meeting_time.day}"
        if c.course.id in lecture_day.get(key, set()):
            penalties += 1.0

    return penalties


# ---------------------------------------------------------------------------
# Registry — maps display name → function
# ---------------------------------------------------------------------------

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