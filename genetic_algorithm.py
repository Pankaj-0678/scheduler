"""
Genetic Algorithm – Performance Optimized
=========================================
Changes over previous version:
- Child SA iterations reduced from 60 → 3 (dramatically faster)
- Heavy SA iterations reduced from 300 → 100
- Conflict-ordered repair now uses random sampling (max 20 combos) instead of exhaustive search
- Optional flag `use_child_sa` to completely disable child SA

Rotation model fix (replaces lab-sync):
- REMOVED: hard constraint that forced all batches of the same lab course to the same time slot.
  That "lab-sync" assumption implemented a parallel-lab model and made rotation schedules
  impossible (the GA would treat correct rotation as a hard conflict and repair it away).
- ADDED: rotation-day constraint — each (section, batch, course) may only appear on ONE
  unique day per week.  If the same lab subject is assigned to the same batch on two
  different days it is penalised as a hard conflict.
- ADDED: rotation-completeness constraint — every batch in a section must be assigned
  at least one session of EVERY lab course belonging to that section.  Missing coverage
  is penalised as a hard conflict so the GA is always pushed toward full coverage.
- REMOVED: all lab-sibling sync propagation from _mutate, _perturb, _conflict_ordered_repair,
  _backtrack_repair and SA._neighbor.  Propagating the same time to all siblings forced
  the parallel model.  Each batch event is now mutated independently.
- REMOVED: _lab_siblings helper (no longer needed anywhere).
"""

from __future__ import annotations
import random
import copy
import math
from typing import List, Optional, Set, Dict, Tuple
from collections import defaultdict
from models import Data, Schedule, ClassEvent


# ======================================================================
# Simulated Annealing
# ======================================================================
class SimulatedAnnealing:
    """Local search optimiser — rotation-model aware."""

    def __init__(self, data: Data, initial_temp: float = 200.0,
                 cooling_rate: float = 0.98, min_temp: float = 0.01,
                 max_iterations: int = 300):
        self.data = data
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations

    def _neighbor(self, schedule: Schedule) -> Schedule:
        """Generate a neighbour by mutating 1–3 individual class events."""
        neighbor = schedule.clone()
        if len(neighbor.classes) < 2:
            return neighbor

        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            mutation_type = random.choice(['time', 'room', 'instructor', 'swap'])

            if mutation_type == 'swap' and len(neighbor.classes) >= 2:
                idx1, idx2 = random.sample(range(len(neighbor.classes)), 2)
                ev1, ev2 = neighbor.classes[idx1], neighbor.classes[idx2]
                # Swap times — each event moves independently (no sibling sync)
                ev1.meeting_time, ev2.meeting_time = ev2.meeting_time, ev1.meeting_time
                if ev1.room.room_type == ev2.room.room_type:
                    ev1.room, ev2.room = ev2.room, ev1.room
            else:
                # Prefer conflicting events
                if neighbor.conflicting_classes:
                    conf_indices = [i for i, ev in enumerate(neighbor.classes)
                                    if ev in neighbor.conflicting_classes]
                    idx = random.choice(conf_indices) if conf_indices else random.randrange(len(neighbor.classes))
                else:
                    idx = random.randrange(len(neighbor.classes))
                ev = neighbor.classes[idx]

                if mutation_type == 'time':
                    ev.meeting_time = random.choice(self.data.meeting_times)
                elif mutation_type == 'room':
                    pool = self.data.lab_rooms if ev.course.course_type == 'Lab' else self.data.lecture_rooms
                    if pool:
                        ev.room = random.choice(pool)
                elif mutation_type == 'instructor':
                    if len(ev.course.instructors) > 1:
                        ev.instructor = random.choice(
                            [i for i in ev.course.instructors if i != ev.instructor]
                        )

        return neighbor

    def _count_hard_conflicts(self, schedule: Schedule) -> int:
        hard = 0
        classes = schedule.classes

        # ── Per-event checks ──────────────────────────────────────────────
        for ev in classes:
            expected = "Lab" if ev.course.course_type == "Lab" else "Lecture"
            if ev.room.room_type != expected:
                hard += 1
            if ev.course.course_type in ("Lecture", "Minor"):
                total_students = ev.section.number_of_students
                if ev.course.course_type == "Minor":
                    for sid in ev.course.enrolled_sections:
                        sec = schedule.data.get_section(sid)
                        if sec:
                            total_students += sec.number_of_students
                if ev.room.capacity < total_students:
                    hard += 1

        # ── Pairwise collisions ───────────────────────────────────────────
        for i in range(len(classes)):
            c1 = classes[i]
            for j in range(i + 1, len(classes)):
                c2 = classes[j]
                if c1.meeting_time.id != c2.meeting_time.id:
                    continue
                instr1 = c1.instructor
                instr2 = c2.instructor
                if instr1 and instr2 and instr1.id == instr2.id:
                    same_lab = (c1.batch != "ALL" and c2.batch != "ALL"
                                and c1.course.id == c2.course.id)
                    if not same_lab:
                        hard += 1
                if c1.room.id == c2.room.id:
                    hard += 1
                if c1.get_affected_section_ids() & c2.get_affected_section_ids():
                    if c1.batch == "ALL" or c2.batch == "ALL":
                        hard += 1
                    elif c1.batch == c2.batch:
                        hard += 1

        # ── Rotation: same lab course must not appear on >1 day per batch ─
        rotation_days: dict[str, set] = {}
        for ev in classes:
            if ev.batch == "ALL":
                continue
            key = f"{ev.section.id}__{ev.batch}__{ev.course.id}"
            rotation_days.setdefault(key, set()).add(ev.meeting_time.day)
        for days in rotation_days.values():
            if len(days) > 1:
                hard += len(days) - 1

        # ── Rotation completeness: every batch must cover all lab courses ─
        for section in schedule.data.sections:
            lab_courses = [
                c for c in schedule.data.section_courses.get(section.id, [])
                if c.course_type == "Lab"
            ]
            if not lab_courses or not section.batches:
                continue
            lab_ids = {c.id for c in lab_courses}
            for batch in section.batches:
                covered = {
                    ev.course.id for ev in classes
                    if ev.section.id == section.id and ev.batch == batch
                }
                hard += len(lab_ids - covered)

        return hard

    def improve(self, schedule: Schedule, verbose: bool = False) -> Schedule:
        current = schedule.clone()
        current_energy = self._count_hard_conflicts(current)
        best = current.clone()
        best_energy = current_energy

        if current_energy == 0:
            return best

        temp = self.initial_temp
        for _ in range(self.max_iterations):
            neighbor = self._neighbor(current)
            neighbor_energy = self._count_hard_conflicts(neighbor)

            delta = neighbor_energy - current_energy
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current = neighbor
                current_energy = neighbor_energy
                if current_energy < best_energy:
                    best = current.clone()
                    best_energy = current_energy
                    if best_energy == 0:
                        break

            temp *= self.cooling_rate
            if temp < self.min_temp:
                break

        best.hard_conflicts = best_energy
        return best


# ======================================================================
# Genetic Algorithm
# ======================================================================
class GeneticAlgorithm:
    def __init__(
        self,
        data: Data,
        active_soft_constraints: dict,
        pop_size: int = 60,
        mutation_rate: float = 0.08,
        tournament_size: int = 4,
        stagnation_limit: int = 25,
        use_sa: bool = True,
        sa_frequency: int = 2,
        sa_iterations: int = 100,
        sa_initial_temp: float = 200.0,
        sa_cooling: float = 0.98,
        restart_threshold: int = 50,
        restart_fraction: float = 0.3,
        use_child_sa: bool = True,
        child_sa_iterations: int = 3,
        child_sa_temp: float = 80.0,
        child_sa_cooling: float = 0.92,
        backtrack_threshold: int = 8,
        max_backtracks: int = 600,
    ):
        self.data = data
        self.active_soft_constraints = active_soft_constraints
        self.pop_size = pop_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.stagnation_limit = stagnation_limit
        self.use_sa = use_sa
        self.sa_frequency = sa_frequency
        self.restart_threshold = restart_threshold
        self.restart_fraction = restart_fraction
        self.backtrack_threshold = backtrack_threshold
        self.max_backtracks = max_backtracks
        self.use_child_sa = use_child_sa

        # Elite SA (heavy)
        self.sa = SimulatedAnnealing(data, initial_temp=sa_initial_temp,
                                     cooling_rate=sa_cooling,
                                     max_iterations=sa_iterations)
        # Child SA (lightweight)
        self.child_sa = SimulatedAnnealing(data,
                                           initial_temp=child_sa_temp,
                                           cooling_rate=child_sa_cooling,
                                           max_iterations=child_sa_iterations)

        self._best_fitness_seen = -1.0
        self._stagnation_counter = 0
        self._generation = 0
        self._no_improvement_counter = 0

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
        self._generation += 1

        # Stagnation detection
        if best.fitness > self._best_fitness_seen + 1e-9:
            self._best_fitness_seen = best.fitness
            self._stagnation_counter = 0
            self._no_improvement_counter = 0
        else:
            self._stagnation_counter += 1
            self._no_improvement_counter += 1

        # Adaptive mutation rate
        ratio = min(1.0, self._stagnation_counter / max(1, self.stagnation_limit))
        self.mutation_rate = min(0.6, self.base_mutation_rate * (1.0 + 5.0 * ratio))

        # Diversity check
        unique_fitness = len({round(s.fitness, 6) for s in population})
        low_diversity = unique_fitness / len(population) < 0.20

        # Population restart
        if self._no_improvement_counter >= self.restart_threshold:
            num_replace = int(self.pop_size * self.restart_fraction)
            for i in range(num_replace):
                fresh = Schedule(self.data)
                self.calculate_fitness(fresh)
                population[-(i + 1)] = fresh
            self._no_improvement_counter = 0
            population.sort(key=lambda s: s.fitness, reverse=True)

        # Elitism
        elite_count = max(2, self.pop_size // 10)
        new_population = [population[i].clone() for i in range(elite_count)]

        # ---- Heavy SA on elite ----
        apply_sa = self.use_sa and (
            self._generation % self.sa_frequency == 0
            or self._stagnation_counter > self.stagnation_limit // 2
            or low_diversity
        )
        if apply_sa:
            for i in range(min(elite_count, len(new_population))):
                if new_population[i].hard_conflicts > 0:
                    improved = self.sa.improve(new_population[i])
                    self.calculate_fitness(improved)
                    new_population[i] = improved

        # ---- Stagnation escape ----
        if self._stagnation_counter >= self.stagnation_limit or low_diversity:

            # 1. Conflict-ordered repair on best (sampled combos)
            repaired = best.clone()
            self._conflict_ordered_repair(repaired)
            self.calculate_fitness(repaired)
            new_population.append(repaired)

            # 2. Backtracking repair
            if best.hard_conflicts <= self.backtrack_threshold * 3:
                bt_repaired = self._backtrack_repair(best.clone())
                self.calculate_fitness(bt_repaired)
                new_population.append(bt_repaired)

            # 3. Heavy perturbation
            perturbed = best.clone()
            self._perturb(perturbed, intensity=0.7)
            self.calculate_fitness(perturbed)
            new_population.append(perturbed)

            if len(population) > 1:
                perturbed2 = population[1].clone()
                self._perturb(perturbed2, intensity=0.5)
                self.calculate_fitness(perturbed2)
                new_population.append(perturbed2)

            # 4. Fresh random schedules
            fresh_count = max(3, self.pop_size // 8)
            for _ in range(fresh_count):
                fresh = Schedule(self.data)
                self.calculate_fitness(fresh)
                new_population.append(fresh)

            if self._stagnation_counter >= self.stagnation_limit:
                self._stagnation_counter = 0

        # ---- Fill rest with crossover + mutation + (optional) child SA ----
        while len(new_population) < self.pop_size:
            if random.random() < 0.2 or self._stagnation_counter > 10:
                parent1 = self._select_diversity(population)
                parent2 = self._select_tournament(population)
            else:
                parent1 = self._select_tournament(population)
                parent2 = self._select_tournament(population)

            child = self._crossover(parent1, parent2)
            self._mutate(child)
            self.calculate_fitness(child)

            if self.use_child_sa and child.hard_conflicts > 0:
                child = self.child_sa.improve(child)
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

    def _select_diversity(self, population: list[Schedule]) -> Schedule:
        lower_half = population[len(population) // 2:]
        return random.choice(lower_half)

    # ------------------------------------------------------------------
    # Conflict-aware crossover
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
        return child

    # ------------------------------------------------------------------
    # Adaptive mutation
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
                # Move this event to a new time — independently, no sibling sync
                ev.meeting_time = random.choice(self.data.meeting_times)
            elif roll < 0.75:
                pool = self.data.lab_rooms if ev.course.course_type == "Lab" else self.data.lecture_rooms
                if pool:
                    ev.room = random.choice(pool)
            else:
                if len(ev.course.instructors) > 1:
                    ev.instructor = random.choice(ev.course.instructors)

    # ------------------------------------------------------------------
    # Helper: per-event pairwise conflict count (for ordering repairs)
    # ------------------------------------------------------------------
    def _event_conflict_counts(self, schedule: Schedule) -> Dict[int, int]:
        counts: Dict[int, int] = {id(ev): 0 for ev in schedule.classes}
        classes = schedule.classes
        for i in range(len(classes)):
            c1 = classes[i]
            for j in range(i + 1, len(classes)):
                c2 = classes[j]
                if c1.meeting_time.id != c2.meeting_time.id:
                    continue
                instr1, instr2 = c1.instructor, c2.instructor
                conflict = False
                if instr1 and instr2 and instr1.id == instr2.id:
                    if not (c1.batch != "ALL" and c2.batch != "ALL"
                            and c1.course.id == c2.course.id):
                        conflict = True
                if c1.room.id == c2.room.id:
                    conflict = True
                if c1.get_affected_section_ids() & c2.get_affected_section_ids():
                    if c1.batch == "ALL" or c2.batch == "ALL" or c1.batch == c2.batch:
                        conflict = True
                if conflict:
                    counts[id(c1)] += 1
                    counts[id(c2)] += 1
        return counts

    # ------------------------------------------------------------------
    # Conflict-ordered repair (sampled combos, not exhaustive)
    # ------------------------------------------------------------------
    def _conflict_ordered_repair(self, schedule: Schedule,
                                 max_passes: int = 80,
                                 sample_size: int = 20) -> None:
        """Repair by trying a limited sample of (time, room) combos per event."""
        self.calculate_fitness(schedule)
        for _ in range(max_passes):
            if schedule.hard_conflicts == 0:
                break
            if not schedule.conflicting_classes:
                break

            counts = self._event_conflict_counts(schedule)
            ev = max(schedule.conflicting_classes,
                     key=lambda e: counts.get(id(e), 0))

            pool = (self.data.lab_rooms if ev.course.course_type == "Lab"
                    else self.data.lecture_rooms) or self.data.rooms

            saved_time = ev.meeting_time
            saved_room = ev.room

            best_hc = schedule.hard_conflicts
            best_time = saved_time
            best_room = saved_room

            all_combos = [(mt, rm) for mt in self.data.meeting_times for rm in pool]
            combos = random.sample(all_combos, min(sample_size, len(all_combos)))

            for mt, rm in combos:
                ev.meeting_time = mt
                ev.room = rm
                self.calculate_fitness(schedule)
                if schedule.hard_conflicts < best_hc:
                    best_hc = schedule.hard_conflicts
                    best_time = mt
                    best_room = rm
                    if best_hc == 0:
                        break

            # Commit best assignment for this event only
            ev.meeting_time = best_time
            ev.room = best_room
            self.calculate_fitness(schedule)

    # ------------------------------------------------------------------
    # Backtracking repair
    # ------------------------------------------------------------------
    def _backtrack_repair(self, schedule: Schedule) -> Schedule:
        self.calculate_fitness(schedule)
        if schedule.hard_conflicts == 0:
            return schedule
        if schedule.hard_conflicts > self.backtrack_threshold:
            self._conflict_ordered_repair(schedule)
            return schedule

        counts = self._event_conflict_counts(schedule)
        ordered: List[ClassEvent] = sorted(
            schedule.conflicting_classes,
            key=lambda e: counts.get(id(e), 0),
            reverse=True
        )
        if not ordered:
            return schedule

        def build_options(ev: ClassEvent) -> List[Tuple]:
            pool = (self.data.lab_rooms if ev.course.course_type == "Lab"
                    else self.data.lecture_rooms) or self.data.rooms
            opts = [(mt, rm) for mt in self.data.meeting_times for rm in pool]
            random.shuffle(opts)
            return opts

        options_map = {id(ev): build_options(ev) for ev in ordered}
        orig_state = {id(ev): (ev.meeting_time, ev.room) for ev in ordered}

        best = schedule.clone()
        best_hc = schedule.hard_conflicts

        stack: List[List[int]] = [[0, 0]]
        backtracks = 0

        while stack and backtracks < self.max_backtracks:
            ev_idx, opt_idx = stack[-1]

            if ev_idx >= len(ordered):
                self.calculate_fitness(schedule)
                if schedule.hard_conflicts < best_hc:
                    best = schedule.clone()
                    best_hc = schedule.hard_conflicts
                    if best_hc == 0:
                        return best
                stack.pop()
                backtracks += 1
                if stack:
                    stack[-1][1] += 1
                continue

            ev = ordered[ev_idx]
            options = options_map[id(ev)]

            if opt_idx >= len(options):
                # Restore this event to its original state and backtrack
                mt0, rm0 = orig_state[id(ev)]
                ev.meeting_time = mt0
                ev.room = rm0
                stack.pop()
                backtracks += 1
                if stack:
                    stack[-1][1] += 1
                continue

            mt, rm = options[opt_idx]
            ev.meeting_time = mt
            ev.room = rm

            self.calculate_fitness(schedule)

            if schedule.hard_conflicts < best_hc:
                best = schedule.clone()
                best_hc = schedule.hard_conflicts
                if best_hc == 0:
                    return best

            if ev not in schedule.conflicting_classes:
                stack.append([ev_idx + 1, 0])
            else:
                stack[-1][1] += 1

        return best

    # ------------------------------------------------------------------
    # Smart perturbation
    # ------------------------------------------------------------------
    def _perturb(self, schedule: Schedule, intensity: float = 0.5) -> None:
        self.calculate_fitness(schedule)
        to_perturb = set(schedule.conflicting_classes)
        non_conf = [ev for ev in schedule.classes if ev not in to_perturb]
        if non_conf:
            extra = max(2, int(len(non_conf) * intensity))
            to_perturb.update(random.sample(non_conf, min(extra, len(non_conf))))

        for ev in to_perturb:
            r = random.random()
            if r < 0.4:
                # Move this event independently — no sibling sync
                ev.meeting_time = random.choice(self.data.meeting_times)
            elif r < 0.7:
                pool = (self.data.lab_rooms if ev.course.course_type == "Lab"
                        else self.data.lecture_rooms)
                if pool:
                    ev.room = random.choice(pool)
            else:
                if len(ev.course.instructors) > 1:
                    ev.instructor = random.choice(ev.course.instructors)

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
                        if sec:
                            total_students += sec.number_of_students
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
                instr1, instr2 = c1.instructor, c2.instructor
                if instr1 and instr2 and instr1.id == instr2.id:
                    # Two different-batch events for the same lab course may share
                    # an instructor (they run in parallel in different rooms).
                    same_lab = (c1.batch != "ALL" and c2.batch != "ALL"
                                and c1.course.id == c2.course.id)
                    if not same_lab:
                        hard_conflicts += 1
                        conflict = True

                if c1.room.id == c2.room.id:
                    hard_conflicts += 1
                    conflict = True

                if c1.get_affected_section_ids() & c2.get_affected_section_ids():
                    # A lecture (batch=ALL) clashes with ANY other event in that
                    # section at the same time — including lab batches.
                    if c1.batch == "ALL" or c2.batch == "ALL":
                        hard_conflicts += 1
                        conflict = True
                    elif c1.batch == c2.batch:
                        # Same batch, same time → clash
                        hard_conflicts += 1
                        conflict = True
                    # Different batches at the same time in the same section → OK
                    # (e.g. B1 in Physics lab, B2 in Chemistry lab simultaneously)

                if conflict:
                    conflicting.add(c1)
                    conflicting.add(c2)

        # ── Rotation: same lab subject must not appear >1 day per batch ───
        # Replacing the old "all batches same time slot" constraint.
        # In a rotation model each (section, batch, course) should be spread
        # across at most one day of the week.
        rotation_days: Dict[str, Set[str]] = defaultdict(set)
        for ev in classes:
            if ev.batch == "ALL":
                continue
            key = f"{ev.section.id}__{ev.batch}__{ev.course.id}"
            rotation_days[key].add(ev.meeting_time.day)

        for key, days in rotation_days.items():
            if len(days) > 1:
                hard_conflicts += len(days) - 1
                parts = key.split("__")
                sec_id, batch, course_id = parts[0], parts[1], parts[2]
                for ev in classes:
                    if (ev.section.id == sec_id and ev.batch == batch
                            and ev.course.id == course_id):
                        conflicting.add(ev)

        # ── Rotation completeness: every batch must cover all lab courses ─
        # If a batch has no session at all for a required lab course that is
        # a hard violation — the schedule is fundamentally incomplete.
        for section in schedule.data.sections:
            lab_courses = [
                c for c in schedule.data.section_courses.get(section.id, [])
                if c.course_type == "Lab"
            ]
            if not lab_courses or not section.batches:
                continue
            lab_ids = {c.id for c in lab_courses}
            for batch in section.batches:
                covered = {
                    ev.course.id for ev in classes
                    if ev.section.id == section.id
                    and ev.batch == batch
                    and ev.course.course_type == "Lab"
                }
                missing_count = len(lab_ids - covered)
                if missing_count > 0:
                    hard_conflicts += missing_count
                    # Mark all events for this batch as conflicting so repair
                    # logic knows where to focus.
                    for ev in classes:
                        if ev.section.id == section.id and ev.batch == batch:
                            conflicting.add(ev)

        total_penalty += hard_conflicts * HARD_PENALTY

        for constr in self.active_soft_constraints.values():
            total_penalty += constr["function"](schedule) * constr["weight"]

        schedule.hard_conflicts = hard_conflicts
        schedule.conflicting_classes = conflicting
        schedule.fitness = 1.0 / (total_penalty + 1.0)