AI Timetable Scheduler: A Genetic Algorithm Approach
This Streamlit application is an advanced, production-ready Master Timetable Generator. It automates the incredibly complex task of scheduling college courses, labs, and minor subjects across multiple divisions, ensuring zero clashes while optimizing for both teacher and student constraints.

It achieves this by modeling the problem as an evolutionary process, using a custom-built Genetic Algorithm.

Core Capabilities
Universal Data Ingestion: Supports importing raw scheduling data directly from CSV files, handling both traditional "University Format" (time-based columns) and "Demo Format" (division-based columns).

Dual-Load Faculty Tracking: Distinctly tracks and enforces separate workload limits for Lecture Hours and Lab Hours for every single instructor.

Dynamic Lab Blocking: Automatically generates contiguous, multi-hour lab sessions (e.g., 2-hour or 3-hour blocks) based on a user-defined slider, intelligently avoiding breaks and lunch hours.

Division-Wise Minor Subjects: Handles the complexities of elective "Minor" subjects (like IoT or Biomedical) by allowing you to assign them to specific, granular student batches rather than entire divisions.

Surgical Conflict Diagnostics: Features a robust "Conflict Inspector" that doesn't just tell you a schedule failed; it tells you exactly why (e.g., "Teacher Lab Overload", "Continuous Lab Blocks", "Room Capacity Exceeded").

Persistent Workspaces: Save your entire university setup (rooms, teachers, courses, divisions) to a .pkl file and load it back instantly in your next session.

How It Works: The Genetic Algorithm (GA)
Scheduling a university isn't just difficult; it is an NP-Hard mathematical problem. If you try to calculate every possible combination of classes, teachers, rooms, and times, it would take a supercomputer millions of years.

Instead of trying every combination, this application uses a Genetic Algorithm—an AI technique inspired by Charles Darwin’s theory of natural selection.

1. Initialization (The Initial Population)
The algorithm starts by generating a "population" of random schedules (e.g., 60 completely random timetables). Most of these will be terrible—teachers will be double-booked, and 200 students will be crammed into a 30-seat lab.

2. Fitness Evaluation (Survival of the Fittest)
Every schedule is graded using a Lexicographic Fitness Function. It evaluates the schedule based on two tiers of rules:

Tier 1: Hard Constraints (Must-Haves)
If any of these are broken, the schedule is fundamentally invalid and receives massive penalties:

Faculty Max Hours: Teachers cannot teach more than their assigned max_lecture_hours or max_lab_hours.

Room Type Match: Lab courses must be in Lab rooms; Lectures in Lecture rooms.

Room Capacity: The room must be large enough to hold the assigned batch or division.

No Overlaps: A teacher, room, or student batch cannot be in two places at the same time.

Lab Rotation Limits: A division cannot have more than 2 lab blocks per day.

No Continuous Labs: Students cannot be forced to take two back-to-back multi-hour lab sessions.

No Continuous Lectures: Teachers get mandatory breaks; they cannot teach continuous back-to-back lectures.

Tier 2: Soft Constraints (Nice-to-Haves)
Once a schedule hits zero Hard Conflicts, the algorithm begins optimizing for Soft Constraints to make the schedule "better":

Respecting teachers' "Morning Preferences."

Minimizing idle gaps for students.

Ensuring smart breaks around lunchtime.

3. Selection & Crossover (Breeding)
The algorithm selects the best-performing schedules (the "parents") and combines them. It takes half of the class placements from Parent A and half from Parent B to create a new "child" schedule, hoping to combine the best traits of both.

4. Smart Heuristic Mutation (Evolution)
To prevent the algorithm from getting stuck, a small percentage of classes are randomly moved to new times or rooms (mutation).

Instead of purely random guessing (which fails for multi-hour labs), the app uses Smart Heuristic Mutation. When it decides to move a 3-hour lab, it scans the timetable, tests a few random slots, and intelligently picks the slot that causes the fewest overlaps.

5. True Greedy Repair
If the algorithm detects that a schedule is "stuck" (stagnating) with just a few stubborn conflicts, it deploys a True Greedy Repair function. It isolates the conflicting class, scans every single valid time and room combination mathematically possible, and forces the class into the absolute best slot.

This cycle of Evaluation, Breeding, and Mutation repeats for hundreds of generations until it produces a perfect, zero-conflict Master Timetable.
