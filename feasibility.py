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

        if self.data.lab_rooms and not self.data.valid_lab_times:
             self._add("ERROR", f"Lab duration is {self.data.lab_duration} hours, but no contiguous blocks of this length exist without hitting lunch/end of day.")

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            lec_events = sum(c.classes_per_week for c in courses if c.course_type in ("Lecture", "Minor", "Minor Lab"))
            lab_courses = [c for c in courses if c.course_type == "Lab"]
            
            total_lab_sessions = sum(c.classes_per_week for c in lab_courses)
            lab_slots_required = max(len(section.batches), total_lab_sessions) * self.data.lab_duration if lab_courses else 0
            
            total_events = lec_events + lab_slots_required

            if total_events > total_slots:
                self._add("ERROR", f"Division '{section.name}' needs {total_events} time slots/week but only {total_slots} exist.")

    def _check_room_sufficiency(self):
        n_sections = len(self.data.sections)
        n_lecture = len(self.data.lecture_rooms)
        n_lab = len(self.data.lab_rooms)

        if n_lecture < n_sections:
            self._add("WARNING", f"Only {n_lecture} lecture room(s) for {n_sections} division(s).")

        if self.data.lab_rooms:
            max_parallel_labs = 0
            for s in self.data.sections:
                lab_courses = [c for c in self.data.section_courses.get(s.id, []) if c.course_type == "Lab"]
                if len(lab_courses) > max_parallel_labs:
                    max_parallel_labs = len(lab_courses)
            if max_parallel_labs > 0 and n_lab < max_parallel_labs:
                self._add("ERROR", f"A section runs {max_parallel_labs} labs in parallel but only {n_lab} lab room(s).")

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

        instr_expected_lec_load = defaultdict(float)
        instr_expected_lab_load = defaultdict(float)

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            for course in courses:
                if not course.instructors: continue
                
                if course.course_type in ("Lecture", "Minor"):
                    events = course.classes_per_week
                    for instr in course.instructors:
                        instr_expected_lec_load[instr.id] += events / len(course.instructors)
                        
                elif course.course_type == "Lab":
                    events = course.classes_per_week * len(section.batches) * self.data.lab_duration
                    for instr in course.instructors:
                        instr_expected_lab_load[instr.id] += events / len(course.instructors)

        for course in self.data.courses:
            if not course.instructors: continue
            if course.course_type == "Minor":
                events = course.classes_per_week
                for instr in course.instructors:
                    instr_expected_lec_load[instr.id] += events / len(course.instructors)
            elif course.course_type == "Minor Lab":
                # PATCH: Safely count all minor batches across all divisions
                total_minor_batches = sum(len(batches) for batches in course.minor_batches.values())
                events = course.classes_per_week * total_minor_batches * self.data.lab_duration
                for instr in course.instructors:
                    instr_expected_lab_load[instr.id] += events / len(course.instructors)

        for instr in self.data.instructors:
            expected_lec = instr_expected_lec_load.get(instr.id, 0)
            expected_lab = instr_expected_lab_load.get(instr.id, 0)
            
            if expected_lec > instr.max_lecture_hours:
                self._add("ERROR", f"Instructor '{instr.name}' is mathematically expected to take {expected_lec:.1f} Lecture hours/week, which exceeds their {instr.max_lecture_hours} hr limit.")
            if expected_lab > instr.max_lab_hours:
                self._add("ERROR", f"Instructor '{instr.name}' is mathematically expected to take {expected_lab:.1f} Lab hours/week, which exceeds their {instr.max_lab_hours} hr limit.")


    def _check_minor_course_refs(self):
        for section in self.data.sections:
            for course in self.data.section_courses.get(section.id, []):
                if not course.course_type.startswith("Minor"): continue
                bad_refs = [sid for sid in course.enrolled_sections if self.data.get_section(sid) is None]
                if bad_refs: self._add("ERROR", f"Minor course references unknown section ID(s).")

    def _check_daily_density(self):
        total_slots = len(self.data.meeting_times)
        slots_per_day = total_slots / 5
        if slots_per_day <= 0: return

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            lec_events = sum(c.classes_per_week for c in courses if c.course_type in ("Lecture", "Minor", "Minor Lab"))
            lab_courses = [c for c in courses if c.course_type == "Lab"]
            
            total_lab_sessions = sum(c.classes_per_week for c in lab_courses)
            lab_slots_required = max(len(section.batches), total_lab_sessions) * self.data.lab_duration if lab_courses else 0
            
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