from __future__ import annotations
from typing import List, Optional, Set, Dict
from collections import defaultdict
import random
import copy

class MeetingTime:
    def __init__(self, id: str, time_str: str, day: str):
        self.id = id
        self.time_str = time_str      
        self.day = day                
        self._hour = int(time_str.split(":")[0])

    @property
    def hour(self) -> int: return self._hour
    @property
    def is_morning(self) -> bool: return self._hour < 12
    @property
    def is_late(self) -> bool: return self._hour >= 15
    def __repr__(self): return f"{self.day} {self.time_str}"

class Room:
    def __init__(self, id: str, capacity: int, room_type: str):
        self.id = id
        self.capacity = capacity
        self.room_type = room_type   

    def __repr__(self): return f"Room({self.id}, {self.room_type})"

class Instructor:
    def __init__(self, id: str, name: str, prefers_morning: bool = False, max_lecture_hours: int = 12, max_lab_hours: int = 12):
        self.id = id
        self.name = name
        self.prefers_morning = prefers_morning
        self.max_lecture_hours = max_lecture_hours
        self.max_lab_hours = max_lab_hours

    def __repr__(self): return f"Instructor({self.name}, Lec={self.max_lecture_hours}, Lab={self.max_lab_hours})"

class Course:
    def __init__(self, id: str, name: str, max_students: int,
                 instructors: List[Instructor], classes_per_week: int,
                 course_type: str, enrolled_sections: List[str] = None):
        self.id = id
        self.name = name
        self.max_students = max_students
        self.instructors = instructors
        self.classes_per_week = classes_per_week
        self.course_type = course_type          
        self.enrolled_sections = enrolled_sections or []
        self.minor_batches = {} 

    def __repr__(self): return f"Course({self.name}, {self.course_type})"

class DepartmentSection:
    def __init__(self, id: str, name: str, number_of_students: int, num_batches: int):
        self.id = id
        self.name = name
        self.number_of_students = number_of_students
        self.num_batches = num_batches
        self.batches = [f"B{b+1}" for b in range(num_batches)]  

    def __repr__(self): return f"Section({self.name}, batches={self.num_batches})"

class Data:
    def __init__(self, rooms: List[Room], meeting_times: List[MeetingTime],
                 instructors: List[Instructor], courses: List[Course],
                 sections: List[DepartmentSection], section_courses: Dict[str, List[Course]], lunch_hour: int, lab_duration: int = 2):
        self.rooms = rooms
        self.meeting_times = meeting_times
        self.instructors = instructors
        self.courses = courses
        self.sections = sections
        self.section_courses = section_courses
        self.lunch_hour = lunch_hour
        self.lab_duration = lab_duration
        self.valid_lab_times = self._calculate_valid_lab_times()

    @property
    def lecture_rooms(self) -> List[Room]: return [r for r in self.rooms if r.room_type == "Lecture"]
    
    @property
    def lab_rooms(self) -> List[Room]: return [r for r in self.rooms if r.room_type == "Lab"]
    
    def get_section(self, section_id: str) -> Optional[DepartmentSection]:
        for s in self.sections:
            if s.id == section_id: return s
        return None

    def get_time_slots(self, start_time: MeetingTime, duration: int) -> List[MeetingTime]:
        slots = [start_time]
        curr = start_time
        for _ in range(1, duration):
            next_slot = next((mt for mt in self.meeting_times if mt.day == curr.day and mt.hour == curr.hour + 1), None)
            if not next_slot: return []
            slots.append(next_slot)
            curr = next_slot
        return slots

    def _calculate_valid_lab_times(self) -> List[MeetingTime]:
        valid_starts = []
        for mt in self.meeting_times:
            if len(self.get_time_slots(mt, self.lab_duration)) == self.lab_duration:
                valid_starts.append(mt)
        return valid_starts

class ClassEvent:
    _counter = 0
    def __init__(self, course: Course, instructor: Instructor, room: Room,
                 start_time: MeetingTime, section: DepartmentSection, data: Data,
                 batch: str = "ALL", sync_group_id: str = None):
        ClassEvent._counter += 1
        self.id = f"CE{ClassEvent._counter}"
        self.course = course
        self.instructor = instructor
        self.room = room
        self.section = section
        self.batch = batch   
        self.sync_group_id = sync_group_id
        
        self.duration = data.lab_duration if "Lab" in course.course_type else 1
        self.time_slots = data.get_time_slots(start_time, self.duration)

    def set_start_time(self, start_time: MeetingTime, data: Data):
        self.time_slots = data.get_time_slots(start_time, self.duration)

    def get_affected_section_ids(self) -> Set[str]:
        # Now strictly isolated to its own section
        return {self.section.id}
            
    def affected_batches(self) -> Set[str]:
        if self.course.course_type.startswith("Minor"):
            if self.course.course_type == "Minor Lab":
                return {self.batch}
            return set(self.course.minor_batches.get(self.section.id, []))
        elif self.batch == "ALL":
            return set(self.section.batches)
        else:
            return {self.batch}

    def __repr__(self): return f"ClassEvent({self.course.name}, {self.section.name}, {self.batch})"

class Schedule:
    def __init__(self, data: Data):
        self.data = data
        self.classes: List[ClassEvent] = []
        self.fitness: float = -1.0
        self.hard_conflicts: int = 0
        self.conflicting_classes: Set[ClassEvent] = set()
        self.conflict_reasons: Dict[ClassEvent, Set[str]] = defaultdict(set)
        self._build_initial_classes()

    def _build_initial_classes(self):
        self.classes = []

        for section in self.data.sections:
            courses = self.data.section_courses.get(section.id, [])
            
            # 1. Process Normal Lectures and Division-Wise Minors
            for course in courses:
                if course.course_type in ("Lecture", "Minor"):
                    batch_label = "ALL" if course.course_type == "Lecture" else "MINOR"
                    for _ in range(course.classes_per_week):
                        ev = self._random_class_event(course, section, batch_label)
                        self.classes.append(ev)
                        
            # 2. Process Parallel Labs and Minor Labs
            lab_courses = [c for c in courses if c.course_type in ("Lab", "Minor Lab")]
            if lab_courses and section.batches:
                total_lab_sessions = sum(c.classes_per_week for c in lab_courses)
                num_slots = max(len(section.batches), total_lab_sessions)
                
                batch_curriculum = []
                for lc in lab_courses:
                    batch_curriculum.extend([lc] * lc.classes_per_week)
                while len(batch_curriculum) < num_slots:
                    batch_curriculum.append(None)
                    
                for slot_idx in range(num_slots):
                    sync_id = f"SYNC_{section.id}_SLOT_{slot_idx}"
                    shared_time = random.choice(self.data.valid_lab_times if self.data.valid_lab_times else self.data.meeting_times)
                    available_lab_rooms = list(self.data.lab_rooms)
                    random.shuffle(available_lab_rooms)
                    
                    busy_instructors = set() 
                    
                    for b_idx, batch in enumerate(section.batches):
                        c_idx = (b_idx + slot_idx) % num_slots
                        course = batch_curriculum[c_idx]
                        
                        if course is None:
                            continue 
                            
                        # Ensure we only schedule a Minor Lab for enrolled batches
                        if course.course_type == "Minor Lab":
                            enrolled_batches = course.minor_batches.get(section.id, [])
                            if batch not in enrolled_batches:
                                continue
                            
                        room = available_lab_rooms.pop() if available_lab_rooms else random.choice(self.data.lab_rooms or self.data.rooms)
                        
                        available_instructors = [ins for ins in course.instructors if ins.id not in busy_instructors]
                        if available_instructors:
                            instructor = random.choice(available_instructors)
                        else:
                            instructor = random.choice(course.instructors)
                            
                        busy_instructors.add(instructor.id)
                        
                        ev = ClassEvent(course, instructor, room, shared_time, section, self.data, batch, sync_group_id=sync_id)
                        self.classes.append(ev)

    def _random_class_event(self, course: Course, section: DepartmentSection, batch: str) -> ClassEvent:
        if not self.data.meeting_times: raise ValueError(f"No meeting times defined")
        if not course.instructors: raise ValueError(f"Course {course.name} has no instructors assigned")
        
        valid_times = self.data.valid_lab_times if "Lab" in course.course_type else self.data.meeting_times
        if not valid_times: valid_times = self.data.meeting_times
        
        meeting_time = random.choice(valid_times)
        instructor = random.choice(course.instructors)
        room_pool = self.data.lab_rooms if "Lab" in course.course_type else self.data.lecture_rooms
        if not room_pool: room_pool = self.data.rooms
        
        room = random.choice(room_pool)
        return ClassEvent(course, instructor, room, meeting_time, section, self.data, batch)

    def clone(self) -> Schedule:
        new = Schedule.__new__(Schedule)
        new.data = self.data
        new.classes = copy.deepcopy(self.classes)
        new.fitness = self.fitness
        new.hard_conflicts = self.hard_conflicts
        
        conf_ids = {c.id for c in self.conflicting_classes}
        new.conflicting_classes = {c for c in new.classes if c.id in conf_ids}
        
        new.conflict_reasons = defaultdict(set)
        for old_ev, reasons in self.conflict_reasons.items():
            for new_ev in new.classes:
                if new_ev.id == old_ev.id:
                    new.conflict_reasons[new_ev] = set(reasons)
                    break

        return new