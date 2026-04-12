from __future__ import annotations
import random
import copy
from typing import List, Set, Dict
from models import Data, Schedule, ClassEvent

class GeneticAlgorithm:
    def __init__(
        self,
        data: Data,
        active_soft_constraints: dict,
        pop_size: int = 60,
        mutation_rate: float = 0.08,
        **kwargs # Catch remaining sidebar kwargs seamlessly
    ):
        self.data = data
        self.active_soft_constraints = active_soft_constraints
        self.pop_size = pop_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = 4
        
        self._stagnation_counter = 0
        self._best_fitness_seen = -1.0

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------
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

        # Elitism
        elite_count = max(2, self.pop_size // 10)
        new_population = [population[i].clone() for i in range(elite_count)]

        # Rapid local optimum breakout (Greedy Target Repair)
        if best.hard_conflicts > 0 and self._stagnation_counter > 5:
            repaired = best.clone()
            self._greedy_conflict_repair(repaired)
            self.calculate_fitness(repaired)
            new_population.append(repaired)
            self._stagnation_counter = 0 # Reset to allow evolution to continue

        # Fill rest with crossover + mutation
        while len(new_population) < self.pop_size:
            parent1 = self._select_tournament(population)
            parent2 = self._select_tournament(population)

            child = self._crossover(parent1, parent2)
            self._mutate(child)
            self.calculate_fitness(child)
            new_population.append(child)

        return new_population

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _select_tournament(self, population: list[Schedule]) -> Schedule:
        k = min(self.tournament_size, len(population))
        tournament = random.sample(population, k)
        return max(tournament, key=lambda s: s.fitness)

    # ------------------------------------------------------------------
    # Crossover (Sync-Aware)
    # ------------------------------------------------------------------
    def _crossover(self, parent1: Schedule, parent2: Schedule) -> Schedule:
        child = Schedule(self.data)
        child.classes = []
        p1_conf = {ev.id for ev in parent1.conflicting_classes}
        p2_conf = {ev.id for ev in parent2.conflicting_classes}

        for i in range(len(parent1.classes)):
            ev1 = parent1.classes[i]
            ev2 = parent2.classes[i]
            in_p1 = ev1.id in p1_conf
            in_p2 = ev2.id in p2_conf

            if in_p1 and not in_p2:
                src = ev2
            elif in_p2 and not in_p1:
                src = ev1
            else:
                src = ev1 if random.random() > 0.5 else ev2

            child.classes.append(copy.deepcopy(src))
            
        # Repair broken sync groups post-crossover
        sync_times = {}
        for ev in child.classes:
            if ev.sync_group_id:
                if ev.sync_group_id not in sync_times:
                    sync_times[ev.sync_group_id] = ev.meeting_time
                else:
                    ev.meeting_time = sync_times[ev.sync_group_id]
                    
        return child

    # ------------------------------------------------------------------
    # MACRO-Mutation
    # ------------------------------------------------------------------
    def _mutate(self, schedule: Schedule) -> None:
        self.calculate_fitness(schedule)
        for ev in schedule.classes:
            is_conflicting = ev in schedule.conflicting_classes
            mut_rate = min(0.95, self.mutation_rate * (2.0 if is_conflicting else 1.0))
            
            if random.random() >= mut_rate:
                continue

            roll = random.random()
            if roll < 0.40:
                # Time mutation: If it's part of a sync group, move the whole group!
                new_mt = random.choice(self.data.meeting_times)
                ev.meeting_time = new_mt
                if ev.sync_group_id:
                    for sibling in schedule.classes:
                        if sibling.sync_group_id == ev.sync_group_id:
                            sibling.meeting_time = new_mt
            elif roll < 0.75:
                # Room mutation
                pool = self.data.lab_rooms if ev.course.course_type == "Lab" else self.data.lecture_rooms
                if pool:
                    ev.room = random.choice(pool)
            else:
                # Instructor mutation
                if len(ev.course.instructors) > 1:
                    ev.instructor = random.choice(ev.course.instructors)

    # ------------------------------------------------------------------
    # Greedy Conflict Repair (Hill Climber to beat Local Optima)
    # ------------------------------------------------------------------
    def _greedy_conflict_repair(self, schedule: Schedule) -> None:
        if not schedule.conflicting_classes: return
        
        # Pick a heavily conflicting event
        target = random.choice(list(schedule.conflicting_classes))
        best_time = target.meeting_time
        min_conflicts = schedule.hard_conflicts

        # Test all timeslots systematically
        for mt in self.data.meeting_times:
            target.meeting_time = mt
            if target.sync_group_id:
                for s in schedule.classes:
                    if s.sync_group_id == target.sync_group_id: 
                        s.meeting_time = mt
            
            self.calculate_fitness(schedule)
            if schedule.hard_conflicts < min_conflicts:
                min_conflicts = schedule.hard_conflicts
                best_time = mt
                if min_conflicts == 0: break

        # Commit best finding
        target.meeting_time = best_time
        if target.sync_group_id:
             for s in schedule.classes:
                 if s.sync_group_id == target.sync_group_id: 
                     s.meeting_time = best_time
        self.calculate_fitness(schedule)

    # ------------------------------------------------------------------
    # Fitness calculation
    # ------------------------------------------------------------------
    def calculate_fitness(self, schedule: Schedule) -> None:
        HARD_PENALTY = 1000
        total_penalty = 0.0
        hard_conflicts = 0
        conflicting: Set[ClassEvent] = set()
        classes = schedule.classes

        # ── Per-event checks ──────────────────────────────────────────────
        for ev in classes:
            expected = "Lab" if ev.course.course_type == "Lab" else "Lecture"
            if ev.room.room_type != expected:
                hard_conflicts += 1
                conflicting.add(ev)

            if ev.course.course_type in ("Lecture", "Minor"):
                total_students = ev.section.number_of_students
                if ev.course.course_type == "Minor":
                    for sid in ev.course.enrolled_sections:
                        sec = schedule.data.get_section(sid)
                        if sec: total_students += sec.number_of_students
                if ev.room.capacity < total_students:
                    hard_conflicts += 1
                    conflicting.add(ev)

        # ── Pairwise collisions ───────────────────────────────────────────
        for i in range(len(classes)):
            c1 = classes[i]
            for j in range(i + 1, len(classes)):
                c2 = classes[j]
                if c1.meeting_time.id != c2.meeting_time.id:
                    continue

                conflict = False
                
                # Instructor overlap
                if c1.instructor and c2.instructor and c1.instructor.id == c2.instructor.id:
                    hard_conflicts += 1
                    conflict = True

                # Room overlap
                if c1.room.id == c2.room.id:
                    hard_conflicts += 1
                    conflict = True

                # Section/Student overlap
                if c1.get_affected_section_ids() & c2.get_affected_section_ids():
                    if c1.batch == "ALL" or c2.batch == "ALL":
                        hard_conflicts += 1
                        conflict = True
                    elif c1.batch == c2.batch:
                        hard_conflicts += 1
                        conflict = True

                if conflict:
                    conflicting.add(c1)
                    conflicting.add(c2)

        # ── Rotation Spacing: Avoid multiple lab blocks on the same day ──
        sync_days = {}
        for ev in classes:
            if ev.sync_group_id:
                key = (ev.section.id, ev.sync_group_id)
                sync_days[key] = ev.meeting_time.day
        
        sec_days = {}
        for (sec_id, sync_id), day in sync_days.items():
            if sec_id not in sec_days: sec_days[sec_id] = []
            sec_days[sec_id].append(day)
            
        for sec_id, days in sec_days.items():
            if len(days) != len(set(days)):
                hard_conflicts += (len(days) - len(set(days))) * 2
                for ev in classes:
                    if ev.section.id == sec_id and ev.sync_group_id:
                        conflicting.add(ev)

        total_penalty += hard_conflicts * HARD_PENALTY

        # Soft Constraints
        for constr in self.active_soft_constraints.values():
            total_penalty += constr["function"](schedule) * constr["weight"]

        schedule.hard_conflicts = hard_conflicts
        schedule.conflicting_classes = conflicting
        schedule.fitness = 1.0 / (total_penalty + 1.0)