from __future__ import annotations
import random
import copy
from collections import defaultdict
from typing import List, Set, Dict
from models import Data, Schedule, ClassEvent

class GeneticAlgorithm:
    def __init__(
        self,
        data: Data,
        active_soft_constraints: dict,
        pop_size: int = 60,
        mutation_rate: float = 0.08,
        **kwargs 
    ):
        self.data = data
        self.active_soft_constraints = active_soft_constraints
        self.pop_size = pop_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = 4
        
        self._stagnation_counter = 0
        self._best_fitness_seen = -1.0

    def initialize_population(self, progress_callback=None) -> list[Schedule]:
        pop = [Schedule(self.data) for _ in range(self.pop_size)]
        for i, s in enumerate(pop):
            self.calculate_fitness(s)
            if progress_callback:
                progress_callback(i + 1, self.pop_size)
        return pop

    def evolve(self, population: list[Schedule]) -> list[Schedule]:
        for s in population:
            if s.fitness < 0:
                self.calculate_fitness(s)

        population.sort(key=lambda s: s.fitness, reverse=True)
        best = population[0]

        if best.fitness > self._best_fitness_seen + 1e-9:
            self._best_fitness_seen = best.fitness
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        elite_count = max(2, self.pop_size // 10)
        new_population = [population[i].clone() for i in range(elite_count)]

        # TRUE GREEDY REPAIR
        if best.hard_conflicts > 0 and self._stagnation_counter > 5:
            repaired = best.clone()
            self._greedy_conflict_repair(repaired)
            self.calculate_fitness(repaired)
            new_population.append(repaired)
        
        # Cataclysmic Reset
        if self._stagnation_counter > 15:
            self._stagnation_counter = 0
            self._best_fitness_seen = best.fitness
            fresh_injection = self.pop_size // 3
            for _ in range(fresh_injection):
                fresh = Schedule(self.data)
                self.calculate_fitness(fresh)
                new_population.append(fresh)

        while len(new_population) < self.pop_size:
            parent1 = self._select_tournament(population)
            parent2 = self._select_tournament(population)

            child = self._crossover(parent1, parent2)
            self._mutate(child)
            self.calculate_fitness(child)
            new_population.append(child)

        return new_population

    def _select_tournament(self, population: list[Schedule]) -> Schedule:
        k = min(self.tournament_size, len(population))
        tournament = random.sample(population, k)
        return max(tournament, key=lambda s: s.fitness)

    def _update_sync_group(self, schedule: Schedule, target: ClassEvent, new_time, new_room, new_inst):
        target.set_start_time(new_time, self.data)
        target.room = new_room
        target.instructor = new_inst
        if target.sync_group_id:
            for s in schedule.classes:
                if s.sync_group_id == target.sync_group_id and s is not target:
                    s.set_start_time(new_time, self.data)

    def _crossover(self, parent1: Schedule, parent2: Schedule) -> Schedule:
        child = Schedule.__new__(Schedule)
        child.data = self.data
        child.fitness = -1.0
        child.hard_conflicts = 0
        child.conflicting_classes = set()
        child.conflict_reasons = defaultdict(set)
        child.classes = []

        p1_conf = {ev.id for ev in parent1.conflicting_classes}
        p2_conf = {ev.id for ev in parent2.conflicting_classes}

        for i in range(len(parent1.classes)):
            ev1 = parent1.classes[i]
            ev2 = parent2.classes[i]
            in_p1 = ev1.id in p1_conf
            in_p2 = ev2.id in p2_conf

            if in_p1 and not in_p2: src = ev2
            elif in_p2 and not in_p1: src = ev1
            else: src = ev1 if random.random() > 0.5 else ev2

            child.classes.append(copy.deepcopy(src))
            
        sync_times = {}
        for ev in child.classes:
            if ev.sync_group_id:
                if ev.sync_group_id not in sync_times:
                    sync_times[ev.sync_group_id] = ev.time_slots[0]
                else:
                    ev.set_start_time(sync_times[ev.sync_group_id], self.data)
                        
        return child

    def _mutate(self, schedule: Schedule) -> None:
        if schedule.fitness < 0:
            self.calculate_fitness(schedule)

        conflicting_ids = {ev.id for ev in schedule.conflicting_classes}

        for ev in schedule.classes:
            is_conflicting = ev.id in conflicting_ids
            # Non-conflicting classes have a much lower chance to mutate, preserving good structures
            mut_rate = min(0.95, self.mutation_rate * (2.0 if is_conflicting else 0.5))
            
            if random.random() >= mut_rate:
                continue

            valid_times = self.data.valid_lab_times if "Lab" in ev.course.course_type else self.data.meeting_times
            if not valid_times: valid_times = self.data.meeting_times
            pool = self.data.lab_rooms if "Lab" in ev.course.course_type else self.data.lecture_rooms
            if not pool: pool = self.data.rooms
            
            new_time = ev.time_slots[0] if ev.time_slots else random.choice(valid_times)
            new_room = ev.room
            new_inst = ev.instructor
            
            roll = random.random()
            
            # SMART HEURISTIC MUTATION
            # Instead of purely dropping multi-hour labs randomly and causing chaos, 
            # this quickly scans 3 random options and selects the safest one.
            if roll < 0.50:
                best_trial_time = random.choice(valid_times)
                best_local_score = 999
                
                for _ in range(3):
                    trial_time = random.choice(valid_times)
                    trial_slots = self.data.get_time_slots(trial_time, ev.duration)
                    if not trial_slots: continue
                    
                    trial_slot_ids = {t.id for t in trial_slots}
                    local_conflicts = 0
                    
                    for other_ev in schedule.classes:
                        if other_ev.id == ev.id: continue
                        other_slot_ids = {t.id for t in other_ev.time_slots}
                        
                        if trial_slot_ids & other_slot_ids:
                            if other_ev.instructor and ev.instructor and other_ev.instructor.id == ev.instructor.id:
                                local_conflicts += 1
                            if other_ev.room.id == ev.room.id:
                                local_conflicts += 1
                                
                    if local_conflicts < best_local_score:
                        best_local_score = local_conflicts
                        best_trial_time = trial_time
                        if best_local_score == 0:
                            break 
                            
                new_time = best_trial_time
                
            elif roll < 0.80:
                new_room = random.choice(pool)
            else:
                if len(ev.course.instructors) > 1:
                    new_inst = random.choice(ev.course.instructors)

            self._update_sync_group(schedule, ev, new_time, new_room, new_inst)

    def _greedy_conflict_repair(self, schedule: Schedule) -> None:
        if not schedule.conflicting_classes:
            return
            
        target = random.choice(list(schedule.conflicting_classes))
        min_conflicts = schedule.hard_conflicts
        
        best_time = target.time_slots[0] if target.time_slots else None
        best_room = target.room
        best_inst = target.instructor
        
        valid_times = self.data.valid_lab_times if "Lab" in target.course.course_type else self.data.meeting_times
        if not valid_times: valid_times = self.data.meeting_times
        pool = self.data.lab_rooms if "Lab" in target.course.course_type else self.data.lecture_rooms
        if not pool: pool = self.data.rooms
        if not valid_times or not pool: return
        
        instructors_to_check = target.course.instructors if len(target.course.instructors) > 1 else [target.instructor]
        
        found_zero = False
        for t_time in valid_times:
            for t_room in pool:
                for t_inst in instructors_to_check:
                    self._update_sync_group(schedule, target, t_time, t_room, t_inst)
                    self.calculate_fitness(schedule)
                    
                    if schedule.hard_conflicts < min_conflicts:
                        min_conflicts = schedule.hard_conflicts
                        best_time = t_time
                        best_room = t_room
                        best_inst = t_inst
                        
                        if min_conflicts == 0:
                            found_zero = True
                            break
                if found_zero: break
            if found_zero: break

        if best_time is not None:
            self._update_sync_group(schedule, target, best_time, best_room, best_inst)
            self.calculate_fitness(schedule)

    def calculate_fitness(self, schedule: Schedule) -> None:
        hard_conflicts = 0
        conflicting: Set[ClassEvent] = set()
        conflict_reasons: Dict[ClassEvent, Set[str]] = defaultdict(set)
        classes = schedule.classes

        # ── 1. Faculty max_hours ─────────────────────────────────────────────
        instructor_lec_hours = defaultdict(int)
        instructor_lab_hours = defaultdict(int)
        
        for ev in classes:
            if ev.instructor:
                for mt in ev.time_slots:
                    if "Lab" in ev.course.course_type:
                        instructor_lab_hours[(ev.instructor.id, mt.id)] += 1
                    else:
                        instructor_lec_hours[(ev.instructor.id, mt.id)] += 1
            
        for instr in self.data.instructors:
            lec_hours = sum(1 for (inst_id, time_id) in instructor_lec_hours.keys() if inst_id == instr.id)
            lab_hours = sum(1 for (inst_id, time_id) in instructor_lab_hours.keys() if inst_id == instr.id)
            
            if lec_hours > instr.max_lecture_hours:
                overload = lec_hours - instr.max_lecture_hours
                hard_conflicts += overload
                for ev in classes:
                    if ev.instructor and ev.instructor.id == instr.id and "Lab" not in ev.course.course_type:
                        conflicting.add(ev)
                        conflict_reasons[ev].add(f"Teacher Lecture Overload ({lec_hours}/{instr.max_lecture_hours})")
                        
            if lab_hours > instr.max_lab_hours:
                overload = lab_hours - instr.max_lab_hours
                hard_conflicts += overload
                for ev in classes:
                    if ev.instructor and ev.instructor.id == instr.id and "Lab" in ev.course.course_type:
                        conflicting.add(ev)
                        conflict_reasons[ev].add(f"Teacher Lab Overload ({lab_hours}/{instr.max_lab_hours})")

        # ── 2. Room type mismatch ─────────────────────────────────────────────
        for ev in classes:
            expected = "Lab" if "Lab" in ev.course.course_type else "Lecture"
            if ev.room.room_type != expected:
                hard_conflicts += 1
                conflicting.add(ev)
                conflict_reasons[ev].add(f"Room Type Mismatch (Needs {expected})")

        # ── 3. Room capacity ──────────────────────────────────────────────────
        for ev in classes:
            if ev.course.course_type not in ("Lecture", "Minor", "Minor Lab"):
                continue

            total_students = 0
            if ev.batch == "ALL":
                total_students = ev.section.number_of_students
            elif ev.batch == "MINOR":
                assigned = ev.course.minor_batches.get(ev.section.id, [])
                stu_per_batch = ev.section.number_of_students // max(1, len(ev.section.batches))
                total_students = len(assigned) * stu_per_batch
            else:
                total_students = ev.section.number_of_students // max(1, len(ev.section.batches))
                    
            if ev.room.capacity < total_students:
                hard_conflicts += 1
                conflicting.add(ev)
                conflict_reasons[ev].add(f"Room Capacity Exceeded ({total_students} > {ev.room.capacity})")

        # ── 4. Overlap detection (room / instructor / section) ────────────────
        for i in range(len(classes)):
            c1 = classes[i]
            c1_times = {t.id for t in c1.time_slots}
            
            for j in range(i + 1, len(classes)):
                c2 = classes[j]
                c2_times = {t.id for t in c2.time_slots}
                
                if not c1_times & c2_times: continue

                conflict = False
                
                if c1.instructor and c2.instructor and c1.instructor.id == c2.instructor.id:
                    hard_conflicts += 1
                    conflict = True
                    conflict_reasons[c1].add("Teacher Double-Booked")
                    conflict_reasons[c2].add("Teacher Double-Booked")

                if c1.room.id == c2.room.id:
                    hard_conflicts += 1
                    conflict = True
                    conflict_reasons[c1].add("Room Double-Booked")
                    conflict_reasons[c2].add("Room Double-Booked")
                        
                if c1.get_affected_section_ids() & c2.get_affected_section_ids():
                    if c1.affected_batches() & c2.affected_batches():
                        hard_conflicts += 1
                        conflict = True
                        conflict_reasons[c1].add("Batch Double-Booked")
                        conflict_reasons[c2].add("Batch Double-Booked")

                if conflict:
                    conflicting.add(c1)
                    conflicting.add(c2)

        # ── 5. Lab rotation slots limit per day ───────────────────
        sync_days: dict = {}
        for ev in classes:
            if ev.sync_group_id and not ev.sync_group_id.startswith("GLOBAL_MINOR"):
                key = (ev.section.id, ev.sync_group_id)
                if key not in sync_days and ev.time_slots:
                    sync_days[key] = ev.time_slots[0].day
        
        sec_day_slots = defaultdict(list)
        for (sec_id, sync_id), day in sync_days.items():
            sec_day_slots[(sec_id, day)].append(sync_id)
            
        for (sec_id, day), sync_ids in sec_day_slots.items():
            if len(sync_ids) > 2: # Limit to 2 lab slots per day (e.g. morning/afternoon)
                overload = len(sync_ids) - 2
                hard_conflicts += overload * 2
                for ev in classes:
                    if ev.section.id == sec_id and ev.sync_group_id in sync_ids:
                        conflicting.add(ev)
                        conflict_reasons[ev].add(f"Too Many Labs on {day} (Max 2 blocks/day)")

        # ── 6. No continuous labs for students ────────────────────────────────
        student_lab_times = defaultdict(list)
        for ev in classes:
            if "Lab" in ev.course.course_type:
                for batch in ev.affected_batches():
                    student_lab_times[(ev.section.id, batch, ev.time_slots[0].day)].append(ev)
        
        for key, lab_evs in student_lab_times.items():
            if len(lab_evs) > 1:
                unique_lab_evs = list({e.id: e for e in lab_evs}.values())
                unique_lab_evs.sort(key=lambda x: x.time_slots[0].hour)
                for i in range(1, len(unique_lab_evs)):
                    prev_end = unique_lab_evs[i-1].time_slots[-1].hour
                    curr_start = unique_lab_evs[i].time_slots[0].hour
                    
                    if curr_start == prev_end + 1:
                        hard_conflicts += 1
                        conflicting.add(unique_lab_evs[i-1])
                        conflicting.add(unique_lab_evs[i])
                        conflict_reasons[unique_lab_evs[i-1]].add("Continuous Lab Blocks")
                        conflict_reasons[unique_lab_evs[i]].add("Continuous Lab Blocks")

        # ── 7. No continuous lectures for teachers ────────────────────────────
        instructor_lec_times = defaultdict(list)
        for ev in classes:
            if ev.instructor and ev.course.course_type in ("Lecture", "Minor"):
                instructor_lec_times[(ev.instructor.id, ev.time_slots[0].day)].append(ev)
                
        for key, lec_evs in instructor_lec_times.items():
            if len(lec_evs) > 1:
                unique_lec_evs = list({e.sync_group_id or e.id: e for e in lec_evs}.values())
                unique_lec_evs.sort(key=lambda x: x.time_slots[0].hour)
                
                for i in range(1, len(unique_lec_evs)):
                    prev_end = unique_lec_evs[i-1].time_slots[-1].hour
                    curr_start = unique_lec_evs[i].time_slots[0].hour
                    
                    if curr_start == prev_end + 1:
                        hard_conflicts += 1
                        conflicting.add(unique_lec_evs[i-1])
                        conflicting.add(unique_lec_evs[i])
                        conflict_reasons[unique_lec_evs[i-1]].add("Continuous Teacher Lectures")
                        conflict_reasons[unique_lec_evs[i]].add("Continuous Teacher Lectures")

        # ── Soft Constraints Evaluation ───────────────────────────────────────
        soft_penalty = 0.0
        for constr in self.active_soft_constraints.values():
            soft_penalty += constr["function"](schedule) * constr["weight"]

        schedule.hard_conflicts = hard_conflicts
        schedule.conflicting_classes = conflicting
        schedule.conflict_reasons = conflict_reasons
        
        # ── LEXICOGRAPHIC FITNESS SCORING ─────────────────────────────────────
        if hard_conflicts > 0:
            # Stage 1: Prioritize ruthlessly eliminating Hard Conflicts. 
            # Output range is (0.0 to 1.0).
            schedule.fitness = 1.0 / (1.0 + hard_conflicts)
        else:
            # Stage 2: Once 0 Hard Conflicts is reached, begin optimizing Soft Constraints.
            # Output range is (1.0 to 2.0). 
            # A schedule with 0 Hard Conflicts will ALWAYS dominate a schedule with 1 Hard Conflict.
            schedule.fitness = 1.0 + (1.0 / (1.0 + soft_penalty))