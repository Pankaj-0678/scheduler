from __future__ import annotations
from typing import List, Optional, Set, Dict, Any
import random
import copy

# ----------------------------------------------------------------------
# MeetingTime
# ----------------------------------------------------------------------
class MeetingTime:
    def __init__(self, id: str, time_str: str, day: str):
        self.id = id
        self.time_str = time_str      # e.g. "09:00 - 10:00"
        self.day = day                # "Mon", "Tue", ...
        self._hour = int(time_str.split(":")[0])

    @property
    def hour(self) -> int:
        return self._hour

    @property
    def is_morning(self) -> bool:
        return self._hour < 12

    @property
    def is_late(self) -> bool:
        return self._hour >= 15

    def __repr__(self):
        return f"{self.day} {self.time_str}"

# ----------------------------------------------------------------------
# Room
# ----------------------------------------------------------------------
class Room:
    def __init__(self, id: str, capacity: int, room_type: str):
        self.id = id
        self.capacity = capacity
        self.room_type = room_type   # "Lecture" or "Lab"

    def __repr__(self):
        return f"Room({self.id}, {self.room_type})"

# ----------------------------------------------------------------------
# Instructor
# ----------------------------------------------------------------------
class Instructor:
    def __init__(self, id: str, name: str, prefers_morning: bool = False):
        self.id = id
        self.name = name
        self.prefers_morning = prefers_morning

    def __repr__(self):
        return f"Instructor({self.name})"

# ----------------------------------------------------------------------
# Course
# ----------------------------------------------------------------------
class Course:
    def __init__(self, id: str, name: str, max_students: int,
                 instructors: List[Instructor], classes_per_week: int,
                 course_type: str, enrolled_sections: List[str] = None):
        self.id = id
        self.name = name
        self.max_students = max_students
        self.instructors = instructors
        self.classes_per_week = classes_per_week
        self.course_type = course_type          # "Lecture", "Lab", "Minor"
        self.enrolled_sections = enrolled_sections or []

    def __repr__(self):
        return f"Course({self.name}, {self.course_type})"

# ----------------------------------------------------------------------
# DepartmentSection (Division)
# ----------------------------------------------------------------------
class DepartmentSection:
    def __init__(self, id: str, name: str, number_of_students: int, num_batches: int):
        self.id = id
        self.name = name
        self.number_of_students = number_of_students
        self.num_batches = num_batches
        self.batches = [f"B{b+1}" for b in range(num_batches)]  # "B1", "B2", ...

    def __repr__(self):
        return f"Section({self.name}, batches={self.num_batches})"

# ----------------------------------------------------------------------
# ClassEvent (a scheduled class)
# ----------------------------------------------------------------------
class ClassEvent:
    _counter = 0

    def __init__(self, course: Course, instructor: Instructor, room: Room,
                 meeting_time: MeetingTime, section: DepartmentSection,
                 batch: str = "ALL", sync_group_id: str = None):
        ClassEvent._counter += 1
        self.id = f"CE{ClassEvent._counter}"
        self.course = course
        self.instructor = instructor
        self.room = room
        self.meeting_time = meeting_time
        self.section = section
        self.batch = batch   # "ALL" for lectures/minors, or "B1", "B2", ... for lab batches
        self.sync_group_id = sync_group_id # Binds parallel rotation labs together

    def get_affected_section_ids(self) -> Set[str]:
        """Return set of section IDs that are busy during this class."""
        if self.course.course_type == "Minor":
            # host section + explicitly enrolled sections
            return {self.section.id} | set(self.course.enrolled_sections)
        else:
            return {self.section.id}

    def __repr__(self):
        return f"ClassEvent({self.course.name}, {self.section.name}, {self.batch})"

# ----------------------------------------------------------------------
# Schedule (a full timetable)
# ----------------------------------------------------------------------
class Schedule:
    def __init__(self, data: Data):
        self.data = data
        self.classes: List[ClassEvent] = []
        self.fitness: float = -1.0
        self.hard_conflicts: int = 0
        self.conflicting_classes: Set[ClassEvent] = set()
        self._build_initial_classes()

    def _build_initial_classes(self):
        """
        Create one ClassEvent for each required teaching slot.
        CRITICAL FIX: Labs are generated as a 'Rotation Matrix'. If a section
        has 4 batches and 4 lab subjects, they are bound together into 4 time slots
        using a sync_group_id.
        """
        self.classes = []
        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            
            # 1. Process Lectures and Minors
            for course in courses:
                if course.course_type in ("Lecture", "Minor"):
                    for _ in range(course.classes_per_week):
                        ev = self._random_class_event(course, section, "ALL")
                        self.classes.append(ev)
            
            # 2. Process Labs (Parallel Rotation Matrix)
            lab_courses = [c for c in courses if c.course_type == "Lab"]
            if lab_courses and section.batches:
                # Calculate required slots based on the maximum dimension
                num_slots = max(len(section.batches), len(lab_courses))
                
                # Each slot gets a unique sync ID so the GA moves them as a single block
                for slot_idx in range(num_slots):
                    sync_id = f"SYNC_{section.id}_SLOT_{slot_idx}"
                    shared_time = random.choice(self.data.meeting_times)
                    
                    # Ensure different rooms for parallel labs
                    available_lab_rooms = list(self.data.lab_rooms)
                    random.shuffle(available_lab_rooms)
                    
                    for b_idx, batch in enumerate(section.batches):
                        # Round-robin subject assignment for rotation
                        c_idx = (b_idx + slot_idx) % len(lab_courses)
                        course = lab_courses[c_idx]
                        
                        room = available_lab_rooms.pop() if available_lab_rooms else random.choice(self.data.lab_rooms or self.data.rooms)
                        instructor = random.choice(course.instructors)
                        
                        ev = ClassEvent(course, instructor, room, shared_time, section, batch, sync_group_id=sync_id)
                        self.classes.append(ev)

    def _random_class_event(self, course: Course, section: DepartmentSection, batch: str) -> ClassEvent:
        """Create a random ClassEvent for non-synced classes."""
        if not self.data.meeting_times:
            raise ValueError(f"No meeting times defined – cannot schedule {course.name}")
        if not course.instructors:
            raise ValueError(f"Course {course.name} has no instructors assigned")
        
        meeting_time = random.choice(self.data.meeting_times)
        instructor = random.choice(course.instructors)
        room_pool = self.data.lab_rooms if course.course_type == "Lab" else self.data.lecture_rooms
        if not room_pool:
            room_pool = self.data.rooms
        
        room = random.choice(room_pool)
        return ClassEvent(course, instructor, room, meeting_time, section, batch)

    def clone(self) -> Schedule:
        """Deep copy of the schedule."""
        new = Schedule.__new__(Schedule)
        new.data = self.data
        new.classes = copy.deepcopy(self.classes)
        new.fitness = self.fitness
        new.hard_conflicts = self.hard_conflicts
        new.conflicting_classes = set(copy.deepcopy(list(self.conflicting_classes)))
        return new

# ----------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------
class Data:
    def __init__(self, rooms: List[Room], meeting_times: List[MeetingTime],
                 instructors: List[Instructor], courses: List[Course],
                 sections: List[DepartmentSection],
                 section_courses: Dict[str, List[Course]],
                 lunch_hour: int):
        self.rooms = rooms
        self.meeting_times = meeting_times
        self.instructors = instructors
        self.courses = courses
        self.sections = sections
        self.section_courses = section_courses
        self.lunch_hour = lunch_hour

    @property
    def lecture_rooms(self) -> List[Room]:
        return [r for r in self.rooms if r.room_type == "Lecture"]

    @property
    def lab_rooms(self) -> List[Room]:
        return [r for r in self.rooms if r.room_type == "Lab"]

    def get_section(self, section_id: str) -> Optional[DepartmentSection]:
        for s in self.sections:
            if s.id == section_id:
                return s
        return None