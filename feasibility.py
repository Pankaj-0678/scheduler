"""
Feasibility Checker for the Timetable Scheduler
================================================
Run this BEFORE starting the GA to detect impossible or near-impossible
scheduling problems and give the user actionable feedback.

Returns a list of (severity, message) tuples:
  "ERROR"   — schedule is mathematically impossible (GA will never converge)
  "WARNING" — schedule is theoretically possible but very tight
  "INFO"    — useful statistics for the user

Usage:
    from feasibility import FeasibilityChecker
    checker = FeasibilityChecker(data)
    results = checker.check()
    for severity, msg in results:
        print(f"[{severity}] {msg}")
"""

from __future__ import annotations
from collections import defaultdict
from models import Data


class FeasibilityChecker:
    """
    Analyses a Data object and reports structural scheduling issues.
    Call .check() to get the full report.
    """

    def __init__(self, data: Data):
        self.data = data
        self._results: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> list[tuple[str, str]]:
        """
        Run all feasibility checks and return (severity, message) list.
        Ordered: ERRORs first, then WARNINGs, then INFOs.
        """
        self._results = []
        self._check_slot_sufficiency()
        self._check_room_sufficiency()
        self._check_instructor_load()
        self._check_minor_course_refs()
        self._check_daily_density()
        self._check_instructor_assignment()
        self._check_lab_batch_room_ratio()

        # Sort: ERRORs first
        order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        self._results.sort(key=lambda x: order.get(x[0], 3))
        return self._results

    def has_errors(self) -> bool:
        return any(sev == "ERROR" for sev, _ in self._results)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_slot_sufficiency(self):
        """
        Each section's total required classes/week must not exceed
        the total number of available time slots.
        """
        total_slots = len(self.data.meeting_times)
        if total_slots == 0:
            self._add("ERROR", "No time slots available. Check college timing settings.")
            return

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            # Count occupied slots from section's perspective.
            # Lab batches all run simultaneously, so they only block 1 slot.
            total_events = 0
            for course in courses:
                if course.course_type in ("Lecture", "Minor"):
                    total_events += course.classes_per_week
                else:
                    # All batches run at the same time in different rooms
                    total_events += course.classes_per_week

            utilization = total_events / total_slots
            if total_events > total_slots:
                self._add(
                    "ERROR",
                    f"Division '{section.name}' needs {total_events} classes/week "
                    f"but only {total_slots} slots exist. The GA cannot produce a "
                    f"conflict-free schedule. Either add more daily hours or reduce courses."
                )
            elif utilization > 0.85:
                self._add(
                    "WARNING",
                    f"Division '{section.name}' uses {utilization:.0%} of available slots "
                    f"({total_events}/{total_slots}). The schedule will be very tight — "
                    f"consider reducing courses or adding time slots."
                )
            else:
                self._add(
                    "INFO",
                    f"Division '{section.name}': {total_events} classes in {total_slots} "
                    f"slots ({utilization:.0%} utilization)."
                )

    def _check_room_sufficiency(self):
        """
        Warn if fewer lecture rooms than sections (all might need rooms simultaneously),
        and error if lab rooms < max simultaneous batches.
        """
        n_sections = len(self.data.sections)
        n_lecture = len(self.data.lecture_rooms)
        n_lab = len(self.data.lab_rooms)

        if n_lecture < n_sections:
            self._add(
                "WARNING",
                f"Only {n_lecture} lecture room(s) for {n_sections} division(s). "
                f"Multiple sections must share time slots — feasible only if schedules "
                f"are staggered. If the GA struggles, add more rooms."
            )

        # Lab rooms must cover the maximum simultaneous batch count
        if self.data.lab_rooms:
            max_batches = max(
                (len(s.batches) for s in self.data.sections
                 if any(c.course_type == "Lab"
                        for c in self.data.section_courses.get(s.id, []))),
                default=0
            )
            if max_batches > 0 and n_lab < max_batches:
                self._add(
                    "ERROR",
                    f"A section has {max_batches} lab batches but only {n_lab} lab room(s). "
                    f"Lab batches run simultaneously and each requires its own room. "
                    f"Add {max_batches - n_lab} more lab room(s)."
                )

    def _check_lab_batch_room_ratio(self):
        """
        Detailed per-section lab room check: simultaneous batches > lab rooms.
        """
        n_lab = len(self.data.lab_rooms)
        if n_lab == 0:
            return  # Already caught by room_sufficiency if labs exist

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            has_lab = any(c.course_type == "Lab" for c in courses)
            if has_lab and len(section.batches) > n_lab:
                self._add(
                    "ERROR",
                    f"Division '{section.name}' has {len(section.batches)} batches "
                    f"but only {n_lab} lab room(s) are available. "
                    f"All batches must run concurrently in separate rooms."
                )

    def _check_instructor_load(self):
        """
        Estimate potential load per instructor.
        If any instructor is exclusively responsible for more events than
        there are time slots, it's impossible.
        """
        total_slots = len(self.data.meeting_times)
        if total_slots == 0:
            return

        # Map: instructor_id -> list of (course, section) pairs
        instr_assignments: dict[str, list] = defaultdict(list)

        for section in self.data.sections:
            for course in self.data.section_courses.get(section.id, []):
                if not course.instructors:
                    continue
                events = course.classes_per_week
                if course.course_type == "Lab":
                    events *= len(section.batches)
                for instr in course.instructors:
                    instr_assignments[instr.id].append(
                        (course.name, section.name, events, len(course.instructors))
                    )

        for instr in self.data.instructors:
            assignments = instr_assignments.get(instr.id, [])
            if not assignments:
                self._add("INFO", f"Instructor '{instr.name}' has no courses assigned.")
                continue

            # Worst-case: instructor teaches ALL occurrences of every assigned course
            max_load = sum(ev for _, _, ev, _ in assignments)
            # Best-case: instructor shares every course equally
            min_load = sum(ev / n_instr for _, _, ev, n_instr in assignments)

            if max_load > total_slots:
                self._add(
                    "WARNING",
                    f"Instructor '{instr.name}' could be assigned up to {max_load} "
                    f"events/week (worst case, sole teacher of all courses). "
                    f"Only {total_slots} slots exist. Ensure co-instructors are assigned."
                )
            elif min_load > total_slots * 0.65:
                self._add(
                    "WARNING",
                    f"Instructor '{instr.name}' has a heavy potential load "
                    f"(~{min_load:.0f} events/week). Consider adding co-instructors."
                )

    def _check_minor_course_refs(self):
        """
        Verify that enrolled_sections in minor courses actually exist,
        and report how many sections are affected.
        """
        for section in self.data.sections:
            for course in self.data.section_courses.get(section.id, []):
                if course.course_type != "Minor":
                    continue

                bad_refs = [
                    sid for sid in course.enrolled_sections
                    if self.data.get_section(sid) is None
                ]
                if bad_refs:
                    self._add(
                        "ERROR",
                        f"Minor course '{course.name}' references unknown section "
                        f"ID(s): {bad_refs}. Fix or remove these references."
                    )
                    continue

                affected = [section] + [
                    self.data.get_section(sid)
                    for sid in course.enrolled_sections
                ]
                section_names = ", ".join(s.name for s in affected)
                self._add(
                    "INFO",
                    f"Minor course '{course.name}' spans {len(affected)} "
                    f"divisions: [{section_names}]. All these divisions must "
                    f"have a free slot at the scheduled time."
                )

                # Check that minor course host section has enough free slots
                total_slots = len(self.data.meeting_times)
                for aff_sec in affected:
                    aff_courses = self.data.section_courses.get(aff_sec.id, [])
                    aff_events = sum(
                        c.classes_per_week for c in aff_courses
                        if c.course_type in ("Lecture", "Lab")
                    )
                    if aff_events >= total_slots - course.classes_per_week:
                        self._add(
                            "WARNING",
                            f"Division '{aff_sec.name}' may have no free slots "
                            f"for minor course '{course.name}'. Its regular load "
                            f"({aff_events} events) nearly fills all {total_slots} slots."
                        )

    def _check_daily_density(self):
        """
        Warn when a section averages more classes per day than slots per day.
        """
        total_slots = len(self.data.meeting_times)
        slots_per_day = total_slots / 5  # 5 working days
        if slots_per_day <= 0:
            return

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            total_events = sum(c.classes_per_week for c in courses)
            avg_per_day = total_events / 5

            if avg_per_day > slots_per_day:
                self._add(
                    "WARNING",
                    f"Division '{section.name}' averages {avg_per_day:.1f} classes/day "
                    f"but there are only {slots_per_day:.1f} slots/day. "
                    f"This is schedulable only if some days are packed."
                )

    def _check_instructor_assignment(self):
        """
        Error if any course has no instructors assigned.
        """
        for section in self.data.sections:
            for course in self.data.section_courses.get(section.id, []):
                if not course.instructors:
                    self._add(
                        "ERROR",
                        f"Course '{course.name}' assigned to division '{section.name}' "
                        f"has NO instructor. Assign at least one."
                    )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _add(self, severity: str, message: str):
        self._results.append((severity, message))