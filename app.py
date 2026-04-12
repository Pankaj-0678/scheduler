"""Master Timetable Generator — Streamlit App (Restored & Rotation Optimized)"""
import io
import random
import streamlit as st
import pandas as pd
 
from models import Data, Room, Instructor, Course, DepartmentSection, MeetingTime
from constraints import SOFT_CONSTRAINTS_REGISTRY
from genetic_algorithm import GeneticAlgorithm
from feasibility import FeasibilityChecker
 
st.set_page_config(page_title="AI Timetable Scheduler", layout="wide")
 
# ── Session state initialisation ──────────────────────────────────────────────
for key, default in [
    ("rooms", []), ("instructors", []), ("courses", []), ("sections", []),
    ("section_courses", {}), ("best_schedule", None), ("run_log", []),
    ("feasibility_results", None),
    ("edit_room_idx", None), ("edit_teacher_idx", None),
    ("edit_course_idx", None), ("edit_section_idx", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_meeting_times(start_hour, end_hour, lunch_hour):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    times = []
    counter = 1
    for day in days:
        for h in range(start_hour, end_hour):
            if h == lunch_hour: continue
            t_str = f"{h:02d}:00 - {h+1:02d}:00"
            times.append(MeetingTime(f"MT{counter}", t_str, day))
            counter += 1
    return times
 
def _reset_data():
    st.session_state.rooms = []
    st.session_state.instructors = []
    st.session_state.courses = []
    st.session_state.sections = []
    st.session_state.section_courses = {}
    st.session_state.best_schedule = None
    st.session_state.run_log = []
    st.session_state.feasibility_results = None
 
def _schedule_to_df(schedule):
    day_order = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5}
    rows = []
    for c in schedule.classes:
        if c.course.course_type == "Minor":
            text = f"📗 {c.course.name} [MINOR]\n{c.instructor.name}\n{c.room.id}"
        elif c.batch == "ALL":
            text = f"📘 {c.course.name}\n{c.instructor.name}\n{c.room.id}"
        else:
            text = f"🔬 {c.course.name}/{c.instructor.name}/{c.batch}/{c.room.id}"
        rows.append({
            "Division": c.section.name,
            "Day": c.meeting_time.day,
            "Time": c.meeting_time.time_str,
            "Event": text,
            "_sort": day_order[c.meeting_time.day],
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    grouped = (df.groupby(["_sort", "Day", "Time", "Division"])["Event"]
               .apply(lambda x: "\n\n".join(x)).reset_index())
    pivot = (grouped.pivot(index=["_sort", "Day", "Time"], columns="Division", values="Event")
             .fillna("---").reset_index().drop(columns=["_sort"]))
    return pivot
 
def _run_feasibility(data):
    checker = FeasibilityChecker(data)
    results = checker.check()
    st.session_state.feasibility_results = results
    return results, checker.has_errors()

def _load_demo_data():
    """Injects a perfect 4-subject / 4-batch rotation environment."""
    _reset_data()
    st.session_state.rooms = [
        Room("Lec1", 60, "Lecture"), Room("Lec2", 60, "Lecture"),
        Room("Lab1", 30, "Lab"), Room("Lab2", 30, "Lab"), 
        Room("Lab3", 30, "Lab"), Room("Lab4", 30, "Lab")
    ]
    st.session_state.instructors = [Instructor(f"T{i}", f"Prof_{i}") for i in range(1, 9)]
    
    c_ai = Course("C1", "AI Theory", 60, [st.session_state.instructors[0]], 3, "Lecture")
    c_ml = Course("C2", "ML Theory", 60, [st.session_state.instructors[1]], 3, "Lecture")
    
    l_ai = Course("L1", "AI Lab", 30, [st.session_state.instructors[4]], 1, "Lab")
    l_ml = Course("L2", "ML Lab", 30, [st.session_state.instructors[5]], 1, "Lab")
    l_dl = Course("L3", "DL Lab", 30, [st.session_state.instructors[6]], 1, "Lab")
    l_wd = Course("L4", "Web Lab", 30, [st.session_state.instructors[7]], 1, "Lab")

    st.session_state.courses = [c_ai, c_ml, l_ai, l_ml, l_dl, l_wd]
    
    sec = DepartmentSection("S1", "Div-A", 60, 4)
    st.session_state.sections = [sec]
    st.session_state.section_courses = {sec.id: st.session_state.courses}
    st.success("✅ Demo Data Loaded! (4 Batches + 4 Lab Subjects setup for perfect rotation)")
 
# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⏰ College Timings")
start_hour  = st.sidebar.slider("Start hour (24h)", 6, 10, 8)
end_hour    = st.sidebar.slider("End hour (24h)", 14, 20, 18)
lunch_hour  = st.sidebar.slider("Lunch start hour", 11, 14, 12)
dynamic_times = _build_meeting_times(start_hour, end_hour, lunch_hour)
 
st.sidebar.markdown("---")
st.sidebar.header("🧬 GA Settings")
generations    = st.sidebar.number_input("Max generations", 50, 1000, 200, 50)
pop_size       = st.sidebar.slider("Population size", 20, 200, 60)
mutation_rate  = st.sidebar.slider("Base mutation rate", 0.01, 0.30, 0.08)
 
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Soft Constraints")
active_constraints = {}
for name, func in SOFT_CONSTRAINTS_REGISTRY.items():
    w = st.sidebar.slider(name, 0.0, 5.0, 1.0, 0.5)
    if w > 0: active_constraints[name] = {"function": func, "weight": w}
 
st.title("🗓️ Master Timetable Generator")
st.caption("GA scheduler with Rotation Synchronisation, Feasibility Checks, and Adaptive Mutation.")
 
tab_import, tab_resources, tab_courses, tab_generate = st.tabs(
    ["📂 Import Data", "🏫 Resources", "📚 Courses & Divisions", "⚙️ Generate Timetable"]
)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Import CSV & Demo Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_import:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Import from CSV")
        st.markdown("Columns: `Day`, `Time`, one column per division.")
    with col2:
        if st.button("🚀 Load Rotation Demo Data"):
            _load_demo_data()

    uploaded_files = st.file_uploader("Upload timetable CSV(s)", type=["csv"], accept_multiple_files=True)
    if uploaded_files and st.button("Parse & Load CSV Data"):
        try:
            rooms_dict, instructors_dict, courses_dict, sections_dict = {}, {}, {}, {}
            section_courses_map, course_week_counts = {}, {}
 
            def _strip_prefix(text: str, prefix: str) -> str:
                if text.startswith(prefix): return text[len(prefix):]
                return text
 
            def _parse_cell(cell_text: str, div_name: str):
                if not cell_text or cell_text.strip() in {"---", "nan", "NaN", ""}: return
                events = cell_text.split("\n\n")
                for raw in events:
                    raw = raw.strip()
                    if not raw or raw == "---": continue
 
                    batch = "ALL"
                    if "🔬" in raw:
                        body  = _strip_prefix(raw, "🔬 ").strip()
                        parts = body.split("/")
                        if len(parts) < 4: continue
                        c_name, t_name, batch, r_id = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                        course_type = "Lab"
                    elif "📗" in raw:
                        body  = _strip_prefix(raw, "📗 ").strip()
                        lines = body.split("\n")
                        if len(lines) < 3: continue
                        c_name = lines[0].replace(" [MINOR]", "").strip()
                        t_name, r_id = lines[1].strip(), lines[2].strip()
                        course_type = "Minor"
                    else:
                        body  = _strip_prefix(raw, "📘 ").strip()
                        lines = body.split("\n")
                        if len(lines) < 3:
                            if len(lines) < 2: continue
                            c_name, t_name, r_id = lines[0].strip(), lines[1].strip(), "R_UNKNOWN"
                        else:
                            c_name, t_name, r_id = lines[0].strip(), lines[1].strip(), lines[2].strip()
                        course_type = "Lecture"
 
                    if not c_name or not t_name: continue
 
                    if r_id not in rooms_dict:
                        rooms_dict[r_id] = Room(r_id, 30 if course_type == "Lab" else 60, "Lab" if course_type == "Lab" else "Lecture")
                    if t_name not in instructors_dict:
                        instructors_dict[t_name] = Instructor(f"T{len(instructors_dict)+1}", t_name)
 
                    c_key = f"{c_name}_{course_type}"
                    if c_key not in courses_dict:
                        courses_dict[c_key] = Course(f"C{len(courses_dict)+1}", c_name, 60 if course_type != "Lab" else 30, [instructors_dict[t_name]], 1, course_type)
                    else:
                        if instructors_dict[t_name] not in courses_dict[c_key].instructors:
                            courses_dict[c_key].instructors.append(instructors_dict[t_name])
 
                    if div_name not in sections_dict:
                        sections_dict[div_name] = DepartmentSection(f"S{len(sections_dict)+1}", div_name, 60, 4)
                        section_courses_map[sections_dict[div_name].id] = set()
 
                    section_courses_map[sections_dict[div_name].id].add(c_key)
                    course_week_counts.setdefault(div_name, {})
                    course_week_counts[div_name][c_key] = course_week_counts[div_name].get(c_key, 0) + 1
 
            for uf in uploaded_files:
                content = uf.getvalue().decode("utf-8-sig")
                df_raw = pd.read_csv(io.StringIO(content))
                df_raw.columns = df_raw.columns.str.strip()
                div_cols  = [c for c in df_raw.columns if c not in {"Unnamed: 0", "Day", "Time"}]
                for col in df_raw.columns: df_raw[col] = df_raw[col].astype(str).str.replace("\r\n", "\n").str.replace("\r", "\n")
                for _, row in df_raw.iterrows():
                    for div in div_cols: _parse_cell(str(row.get(div, "")), div)
 
            for div, counts in course_week_counts.items():
                for c_key, cnt in counts.items():
                    if c_key in courses_dict: courses_dict[c_key].classes_per_week = max(courses_dict[c_key].classes_per_week, cnt)
 
            _reset_data()
            st.session_state.rooms = list(rooms_dict.values())
            st.session_state.instructors = list(instructors_dict.values())
            st.session_state.courses = list(courses_dict.values())
            st.session_state.sections = list(sections_dict.values())
            st.session_state.section_courses = {s_id: [courses_dict[ck] for ck in ck_set if ck in courses_dict] for s_id, ck_set in section_courses_map.items()}
            st.success(f"✅ Loaded: {len(st.session_state.rooms)} rooms, {len(st.session_state.instructors)} instructors, {len(st.session_state.courses)} courses, {len(st.session_state.sections)} divisions.")
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 & 2 — Resources and Courses (Abridged for brevity, assuming standard Add/Edit)
# ══════════════════════════════════════════════════════════════════════════════
with tab_resources:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Rooms")
        with st.form("room_form", clear_on_submit=True):
            r_name = st.text_input("Room name")
            r_type = st.radio("Room type", ["Lecture", "Lab"], horizontal=True)
            if st.form_submit_button("➕ Add Room") and r_name.strip():
                st.session_state.rooms.append(Room(r_name.strip(), 60, r_type)); st.success(f"Added {r_name}")
        for i, r in enumerate(st.session_state.rooms): st.write(f"• **{r.id}** — {r.room_type}")
 
    with col2:
        st.subheader("👩‍🏫 Teachers")
        with st.form("teacher_form", clear_on_submit=True):
            t_name = st.text_input("Teacher name")
            if st.form_submit_button("➕ Add Teacher") and t_name.strip():
                st.session_state.instructors.append(Instructor(f"T{len(st.session_state.instructors)+1}", t_name.strip())); st.success(f"Added {t_name}")
        for i, t in enumerate(st.session_state.instructors): st.write(f"• **{t.name}**")

with tab_courses:
    st.subheader("📚 Courses")
    with st.form("course_form", clear_on_submit=True):
        c_name = st.text_input("Course name")
        c_type = st.radio("Course type", ["Lecture", "Lab", "Minor"], horizontal=True)
        c_weeks = st.number_input("Classes per week", 1, 10, 3)
        sel_teachers = st.multiselect("Assign teachers", [t.name for t in st.session_state.instructors])
        if st.form_submit_button("➕ Add Course") and c_name.strip() and sel_teachers:
            assigned = [t for t in st.session_state.instructors if t.name in sel_teachers]
            st.session_state.courses.append(Course(f"C{len(st.session_state.courses)+1}", c_name.strip(), 60, assigned, c_weeks, c_type))
            st.success(f"Added {c_name}")
    for c in st.session_state.courses: st.write(f"• **{c.name}** ({c.course_type}) — {c.classes_per_week}/wk")
 
    st.subheader("🏢 Divisions (Sections)")
    with st.form("section_form", clear_on_submit=True):
        s_name = st.text_input("Division name")
        s_batches = st.number_input("Number of lab batches", 1, 10, 4)
        sel_courses = st.multiselect("Select courses for this division", [c.name for c in st.session_state.courses])
        if st.form_submit_button("➕ Add Division") and s_name.strip() and sel_courses:
            s_id = f"S{len(st.session_state.sections)+1}"
            st.session_state.sections.append(DepartmentSection(s_id, s_name.strip(), 60, s_batches))
            st.session_state.section_courses[s_id] = [c for c in st.session_state.courses if c.name in sel_courses]
            st.success(f"Added division: {s_name}")
    for s in st.session_state.sections: st.write(f"• **{s.name}** — {len(s.batches)} batches")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Generate Timetable (RESTORED FEASIBILITY & ERROR HANDLING)
# ══════════════════════════════════════════════════════════════════════════════
with tab_generate:
    st.subheader("Generate Master Timetable")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rooms",     len(st.session_state.rooms))
    c2.metric("Teachers",  len(st.session_state.instructors))
    c3.metric("Courses",   len(st.session_state.courses))
    c4.metric("Divisions", len(st.session_state.sections))
 
    # RESTORED: Step 1 — Feasibility Check
    st.markdown("### Step 1 — Feasibility Check")
    if st.button("🔍 Run Feasibility Check"):
        if not st.session_state.sections:
            st.error("No divisions added.")
        else:
            data_tmp = Data(
                st.session_state.rooms, dynamic_times,
                st.session_state.instructors, st.session_state.courses,
                st.session_state.sections, st.session_state.section_courses,
                lunch_hour,
            )
            results, has_errors = _run_feasibility(data_tmp)
            error_count   = sum(1 for s, _ in results if s == "ERROR")
            warning_count = sum(1 for s, _ in results if s == "WARNING")
            col_e, col_w, col_i = st.columns(3)
            col_e.metric("❌ Errors",   error_count)
            col_w.metric("⚠️ Warnings", warning_count)
            col_i.metric("ℹ️ Info",     len(results) - error_count - warning_count)
            if has_errors:
                st.error("Critical issues found – fix errors before generating.")
            elif warning_count:
                st.warning("Warnings found – schedule may be tight.")
            else:
                st.success("Ready to generate!")
 
    if st.session_state.feasibility_results:
        with st.expander("Feasibility Report", expanded=True):
            for sev, msg in st.session_state.feasibility_results:
                if sev == "ERROR": st.error(f"❌ ERROR: {msg}")
                elif sev == "WARNING": st.warning(f"⚠️ WARNING: {msg}")
                else: st.info(f"ℹ️ {msg}")
 
    st.markdown("---")
    st.markdown("### Step 2 — Generate")
    if st.button("🚀 Generate Master Timetable", type="primary"):
        errors = []
        if not st.session_state.rooms:    errors.append("No rooms added.")
        if not st.session_state.courses:  errors.append("No courses added.")
        if not st.session_state.sections: errors.append("No divisions added.")
        if errors:
            for e in errors: st.error(e)
            st.stop()
 
        # RESTORED: Try/Except Safety Net
        try:
            data = Data(
                st.session_state.rooms, dynamic_times,
                st.session_state.instructors, st.session_state.courses,
                st.session_state.sections, st.session_state.section_courses,
                lunch_hour,
            )
 
            ga = GeneticAlgorithm(
                data, active_constraints,
                pop_size=pop_size,
                mutation_rate=mutation_rate,
            )
 
            progress_bar  = st.progress(0, text="Initialising…")
            status_text   = st.empty()
            
            try:
                population = ga.initialize_population()
            except Exception as init_err:
                st.error(f"❌ Failed to initialise population: {init_err}")
                st.stop()
 
            best_schedule = None
 
            for gen in range(int(generations)):
                try:
                    population = ga.evolve(population)
                except Exception as evolve_err:
                    st.error(f"❌ Error at generation {gen+1}: {evolve_err}")
                    break
 
                best_schedule = population[0]
                pct = (gen + 1) / int(generations)
                progress_bar.progress(
                    pct,
                    text=f"Generation {gen+1}/{int(generations)} | "
                         f"fitness={best_schedule.fitness:.4f} | "
                         f"conflicts={best_schedule.hard_conflicts}",
                )
 
                if best_schedule.hard_conflicts == 0 and gen > 15:
                    progress_bar.progress(1.0, text=f"Early stop — converged in {gen+1} generations!")
                    break
 
            if best_schedule is not None:
                st.session_state.best_schedule = best_schedule
            else:
                st.error("No schedule produced.")
 
        except Exception as outer_err:
            st.error(f"❌ Unexpected error: {outer_err}")
 
    # ── Display result ────────────────────────────────────────────────────────
    best = st.session_state.best_schedule
    if best is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Final Fitness",  f"{best.fitness:.6f}")
        col_b.metric("Hard Conflicts", best.hard_conflicts,
                     delta=None if best.hard_conflicts > 0 else "✅ Zero")
        col_c.metric("Total Classes",  len(best.classes))
        
        if best.hard_conflicts > 0:
            st.warning("Schedule has conflicts. Try generating again or check Feasibility.")
        else:
            st.success("Perfect schedule – zero hard conflicts!")
 
        st.subheader("Master Timetable Matrix")
        df_pivot = _schedule_to_df(best)
        st.dataframe(df_pivot, use_container_width=True, height=800)
 
        st.subheader("Per-Division Timetable")
        div_names = sorted({c.section.name for c in best.classes})
        if div_names:
            selected_div = st.selectbox("Select division", div_names)
            div_rows = []
            for c in best.classes:
                if c.section.name != selected_div: continue
                if c.course.course_type == "Minor":
                    text = f"📗 {c.course.name} [MINOR] | {c.instructor.name} | {c.room.id}"
                elif c.batch == "ALL":
                    text = f"📘 {c.course.name} | {c.instructor.name} | {c.room.id}"
                else:
                    text = f"🔬 {c.course.name} | {c.instructor.name} | {c.batch} | {c.room.id}"
                div_rows.append({
                    "Day": c.meeting_time.day,
                    "Time": c.meeting_time.time_str,
                    "Event": text,
                    "_sort": {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5}[c.meeting_time.day],
                })
            if div_rows:
                df_div = pd.DataFrame(div_rows).sort_values(["_sort", "Time"]).drop(columns=["_sort"])
                st.dataframe(df_div, use_container_width=True, height=500)
 
        # RESTORED: Conflict Inspector
        if best.hard_conflicts > 0:
            st.subheader("🔴 Conflict Inspector")
            conflict_rows = []
            for ev in best.conflicting_classes:
                conflict_rows.append({
                    "Division": ev.section.name,
                    "Course":   ev.course.name,
                    "Type":     ev.course.course_type,
                    "Batch":    ev.batch,
                    "Teacher":  ev.instructor.name,
                    "Room":     ev.room.id,
                    "Day":      ev.meeting_time.day,
                    "Time":     ev.meeting_time.time_str,
                })
            if conflict_rows:
                st.dataframe(pd.DataFrame(conflict_rows), use_container_width=True)