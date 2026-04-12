from __future__ import annotations
from collections import defaultdict
from models import Data

class FeasibilityChecker:
    def __init__(self, data: Data):
        self.data = data
        self._results: list[tuple[str, str]] = []

    def check(self) -> list[tuple[str, str]]:
        self._results = []
        self._check_slot_sufficiency()
        self._check_room_sufficiency()
        self._check_instructor_load()
        self._check_minor_course_refs()
        self._check_daily_density()
        self._check_instructor_assignment()
        self._check_lab_batch_room_ratio()

        order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        self._results.sort(key=lambda x: order.get(x[0], 3))
        return self._results

    def has_errors(self) -> bool:
        return any(sev == "ERROR" for sev, _ in self._results)

    def _check_slot_sufficiency(self):
        total_slots = len(self.data.meeting_times)
        if total_slots == 0:
            self._add("ERROR", "No time slots available. Check college timing settings.")
            return

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            
            lec_events = sum(c.classes_per_week for c in courses if c.course_type in ("Lecture", "Minor"))
            lab_courses = [c for c in courses if c.course_type == "Lab"]
            
            # Labs run in a parallel matrix based on max(batches, subjects)
            lab_slots_required = max(len(section.batches), len(lab_courses)) if lab_courses else 0
            
            total_events = lec_events + lab_slots_required

            utilization = total_events / total_slots
            if total_events > total_slots:
                self._add("ERROR", f"Division '{section.name}' needs {total_events} classes/week but only {total_slots} slots exist.")
            elif utilization > 0.85:
                self._add("WARNING", f"Division '{section.name}' uses {utilization:.0%} of available slots.")

    def _check_room_sufficiency(self):
        n_sections = len(self.data.sections)
        n_lecture = len(self.data.lecture_rooms)
        n_lab = len(self.data.lab_rooms)

        if n_lecture < n_sections:
            self._add("WARNING", f"Only {n_lecture} lecture room(s) for {n_sections} division(s).")

        if self.data.lab_rooms:
            max_batches = max((len(s.batches) for s in self.data.sections if any(c.course_type == "Lab" for c in self.data.section_courses.get(s.id, []))), default=0)
            if max_batches > 0 and n_lab < max_batches:
                self._add("ERROR", f"A section has {max_batches} lab batches but only {n_lab} lab room(s).")

    def _check_lab_batch_room_ratio(self):
        n_lab = len(self.data.lab_rooms)
        if n_lab == 0: return

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            has_lab = any(c.course_type == "Lab" for c in courses)
            if has_lab and len(section.batches) > n_lab:
                self._add("ERROR", f"Division '{section.name}' has {len(section.batches)} batches but only {n_lab} lab room(s).")

    def _check_instructor_load(self):
        total_slots = len(self.data.meeting_times)
        if total_slots == 0: return

        instr_assignments = defaultdict(list)
        for section in self.data.sections:
            for course in self.data.section_courses.get(section.id, []):
                if not course.instructors: continue
                # Labs now only count as 1 slot per week against an instructor's capacity if rotation is active
                events = course.classes_per_week 
                for instr in course.instructors:
                    instr_assignments[instr.id].append((course.name, section.name, events, len(course.instructors)))

        for instr in self.data.instructors:
            assignments = instr_assignments.get(instr.id, [])
            if not assignments: continue
            max_load = sum(ev for _, _, ev, _ in assignments)
            if max_load > total_slots:
                self._add("WARNING", f"Instructor '{instr.name}' could be assigned up to {max_load} events/week. Only {total_slots} exist.")

    def _check_minor_course_refs(self):
        for section in self.data.sections:
            for course in self.data.section_courses.get(section.id, []):
                if course.course_type != "Minor": continue
                bad_refs = [sid for sid in course.enrolled_sections if self.data.get_section(sid) is None]
                if bad_refs: self._add("ERROR", f"Minor course references unknown section ID(s).")

    def _check_daily_density(self):
        total_slots = len(self.data.meeting_times)
        slots_per_day = total_slots / 5
        if slots_per_day <= 0: return

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            lec_events = sum(c.classes_per_week for c in courses if c.course_type in ("Lecture", "Minor"))
            lab_courses = [c for c in courses if c.course_type == "Lab"]
            lab_slots_required = max(len(section.batches), len(lab_courses)) if lab_courses else 0
            
            avg_per_day = (lec_events + lab_slots_required) / 5
            if avg_per_day > slots_per_day:
                self._add("WARNING", f"Division '{section.name}' averages {avg_per_day:.1f} classes/day but there are only {slots_per_day:.1f} slots.")

    def _check_instructor_assignment(self):
        for section in self.data.sections:
            for course in self.data.section_courses.get(section.id, []):
                if not course.instructors:
                    self._add("ERROR", f"Course '{course.name}' has NO instructor.")

    def _add(self, severity: str, message: str):
        self._results.append((severity, message))