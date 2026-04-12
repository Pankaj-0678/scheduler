"""Master Timetable Generator — Streamlit App (Fixed)"""
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
    # edit state keys
    ("edit_room_idx", None), ("edit_teacher_idx", None),
    ("edit_course_idx", None), ("edit_section_idx", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default
 
# ── Helper: build meeting times ───────────────────────────────────────────────
def _build_meeting_times(start_hour, end_hour, lunch_hour):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    times = []
    counter = 1
    for day in days:
        for h in range(start_hour, end_hour):
            if h == lunch_hour:
                continue
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
    if df.empty:
        return df
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
tournament_sz  = st.sidebar.slider("Tournament size", 2, 10, 4)
stagnation_lim = st.sidebar.slider("Stagnation limit", 10, 60, 25)
 
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Soft Constraints")
active_constraints = {}
for name, func in SOFT_CONSTRAINTS_REGISTRY.items():
    w = st.sidebar.slider(name, 0.0, 5.0, 1.0, 0.5)
    if w > 0:
        active_constraints[name] = {"function": func, "weight": w}
 
st.title("🗓️ Master Timetable Generator")
st.caption("GA scheduler with feasibility checking, minor courses, adaptive mutation.")
 
tab_import, tab_resources, tab_courses, tab_generate = st.tabs(
    ["📂 Import Data", "🏫 Resources", "📚 Courses & Divisions", "⚙️ Generate Timetable"]
)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Import CSV
# Fixed: emoji prefix stripping now uses str.removeprefix / replace instead of
# lstrip (which strips individual characters, not substrings).
# Also handles cells that have no emoji prefix (plain "COURSE\nTEACHER\nROOM").
# Supports multiple uploaded files in one go.
# ══════════════════════════════════════════════════════════════════════════════
with tab_import:
    st.subheader("Import from CSV")
    st.markdown(
        "Upload **one or more** timetable CSVs. "
        "Columns: `Day`, `Time`, one column per division. "
        "Cell formats: `📘 COURSE\\nTEACHER\\nROOM`, "
        "`🔬 COURSE/TEACHER/BATCH/ROOM`, `📗 COURSE [MINOR]\\nTEACHER\\nROOM`."
    )
 
    uploaded_files = st.file_uploader(
        "Upload timetable CSV(s)", type=["csv"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Parse & Load CSV Data"):
        try:
            rooms_dict        = {}
            instructors_dict  = {}
            courses_dict      = {}
            sections_dict     = {}
            section_courses_map = {}
            course_week_counts  = {}  # {div_name: {c_key: count}}
 
            def _strip_prefix(text: str, prefix: str) -> str:
                """Remove emoji+space prefix safely (works regardless of byte width)."""
                if text.startswith(prefix):
                    return text[len(prefix):]
                return text
 
            def _parse_cell(cell_text: str, div_name: str):
                if not cell_text or cell_text.strip() in {"---", "nan", "NaN", ""}:
                    return
                events = cell_text.split("\n\n")
                for raw in events:
                    raw = raw.strip()
                    if not raw or raw == "---":
                        continue
 
                    batch = "ALL"
 
                    if "🔬" in raw:
                        # Format: 🔬 COURSE/TEACHER/BATCH/ROOM
                        body  = _strip_prefix(raw, "🔬 ").strip()
                        parts = body.split("/")
                        if len(parts) < 4:
                            continue
                        c_name = parts[0].strip()
                        t_name = parts[1].strip()
                        batch  = parts[2].strip()
                        r_id   = parts[3].strip()
                        course_type = "Lab"
 
                    elif "📗" in raw:
                        # Format: 📗 COURSE [MINOR]\nTEACHER\nROOM
                        body  = _strip_prefix(raw, "📗 ").strip()
                        lines = body.split("\n")
                        if len(lines) < 3:
                            continue
                        c_name = lines[0].replace(" [MINOR]", "").strip()
                        t_name = lines[1].strip()
                        r_id   = lines[2].strip()
                        course_type = "Minor"
 
                    else:
                        # Lecture (📘 prefix optional)
                        body  = _strip_prefix(raw, "📘 ").strip()
                        lines = body.split("\n")
                        if len(lines) < 3:
                            # Try 2-line fallback (no room info)
                            if len(lines) < 2:
                                continue
                            c_name = lines[0].strip()
                            t_name = lines[1].strip()
                            r_id   = "R_UNKNOWN"
                        else:
                            c_name = lines[0].strip()
                            t_name = lines[1].strip()
                            r_id   = lines[2].strip()
                        course_type = "Lecture"
 
                    if not c_name or not t_name:
                        continue
 
                    # Room
                    if r_id not in rooms_dict:
                        cap    = 30 if course_type == "Lab" else 60
                        r_type = "Lab" if course_type == "Lab" else "Lecture"
                        rooms_dict[r_id] = Room(r_id, cap, r_type)
 
                    # Instructor
                    if t_name not in instructors_dict:
                        t_id = f"T{len(instructors_dict)+1}"
                        instructors_dict[t_name] = Instructor(t_id, t_name)
 
                    # Course
                    c_key = f"{c_name}_{course_type}"
                    if c_key not in courses_dict:
                        c_id = f"C{len(courses_dict)+1}"
                        courses_dict[c_key] = Course(
                            c_id, c_name,
                            max_students=60 if course_type != "Lab" else 30,
                            instructors=[instructors_dict[t_name]],
                            classes_per_week=1,
                            course_type=course_type,
                        )
                    else:
                        existing = courses_dict[c_key]
                        t_obj = instructors_dict[t_name]
                        if t_obj not in existing.instructors:
                            existing.instructors.append(t_obj)
 
                    # Section
                    if div_name not in sections_dict:
                        s_id = f"S{len(sections_dict)+1}"
                        sections_dict[div_name] = DepartmentSection(
                            s_id, div_name, number_of_students=60, num_batches=4
                        )
                        section_courses_map[s_id] = set()
 
                    s_id = sections_dict[div_name].id
                    section_courses_map[s_id].add(c_key)
 
                    # Week count
                    course_week_counts.setdefault(div_name, {})
                    course_week_counts[div_name][c_key] = (
                        course_week_counts[div_name].get(c_key, 0) + 1
                    )
 
            # ── process each uploaded file ──────────────────────────────────
            for uf in uploaded_files:
                content = uf.getvalue().decode("utf-8-sig")
                from io import StringIO
                df_raw = pd.read_csv(StringIO(content))
                df_raw.columns = df_raw.columns.str.strip()
                meta_cols = {"Unnamed: 0", "Day", "Time"}
                div_cols  = [c for c in df_raw.columns if c not in meta_cols]
 
                for col in df_raw.columns:
                    df_raw[col] = (df_raw[col].astype(str)
                                   .str.replace("\r\n", "\n")
                                   .str.replace("\r", "\n"))
 
                for _, row in df_raw.iterrows():
                    for div in div_cols:
                        _parse_cell(str(row.get(div, "")), div)
 
            # Update classes_per_week
            for div, counts in course_week_counts.items():
                for c_key, cnt in counts.items():
                    if c_key in courses_dict:
                        courses_dict[c_key].classes_per_week = max(
                            courses_dict[c_key].classes_per_week, cnt
                        )
 
            _reset_data()
            st.session_state.rooms       = list(rooms_dict.values())
            st.session_state.instructors = list(instructors_dict.values())
            st.session_state.courses     = list(courses_dict.values())
            st.session_state.sections    = list(sections_dict.values())
            st.session_state.section_courses = {
                s_id: [courses_dict[ck] for ck in ck_set if ck in courses_dict]
                for s_id, ck_set in section_courses_map.items()
            }
 
            st.success(
                f"✅ Loaded from {len(uploaded_files)} file(s): "
                f"{len(st.session_state.rooms)} rooms, "
                f"{len(st.session_state.instructors)} instructors, "
                f"{len(st.session_state.courses)} courses, "
                f"{len(st.session_state.sections)} divisions."
            )
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            st.exception(e)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Resources (Rooms & Teachers) with Add / Edit / Remove
# ══════════════════════════════════════════════════════════════════════════════
with tab_resources:
    col1, col2 = st.columns(2)
 
    # ── Rooms ─────────────────────────────────────────────────────────────────
    with col1:
        st.subheader("🏠 Rooms")
        with st.form("room_form", clear_on_submit=True):
            r_name = st.text_input("Room name")
            r_type = st.radio("Room type", ["Lecture", "Lab"], horizontal=True)
            r_cap  = st.number_input("Capacity", 10, 300, 60)
            if st.form_submit_button("➕ Add Room") and r_name.strip():
                # Prevent duplicate IDs
                existing_ids = {r.id for r in st.session_state.rooms}
                if r_name.strip() in existing_ids:
                    st.error("A room with that name already exists.")
                else:
                    st.session_state.rooms.append(Room(r_name.strip(), r_cap, r_type))
                    st.success(f"Added {r_type} room: {r_name}")
 
        if st.session_state.rooms:
            st.write("**Rooms:**")
            for i, r in enumerate(st.session_state.rooms):
                c_left, c_edit, c_del = st.columns([5, 1, 1])
                c_left.write(f"• **{r.id}** — {r.room_type} (cap {r.capacity})")
                if c_edit.button("✏️", key=f"edit_room_{i}"):
                    st.session_state.edit_room_idx = i
                if c_del.button("🗑️", key=f"del_room_{i}"):
                    st.session_state.rooms.pop(i)
                    if st.session_state.edit_room_idx == i:
                        st.session_state.edit_room_idx = None
                    st.rerun()
 
            # Inline edit form
            ei = st.session_state.edit_room_idx
            if ei is not None and ei < len(st.session_state.rooms):
                er = st.session_state.rooms[ei]
                st.markdown("**Edit Room:**")
                with st.form("edit_room_form"):
                    new_cap  = st.number_input("Capacity", 10, 300, int(er.capacity))
                    new_type = st.radio("Type", ["Lecture", "Lab"],
                                        index=0 if er.room_type == "Lecture" else 1,
                                        horizontal=True)
                    c_save, c_cancel = st.columns(2)
                    if c_save.form_submit_button("💾 Save"):
                        st.session_state.rooms[ei].capacity  = new_cap
                        st.session_state.rooms[ei].room_type = new_type
                        st.session_state.edit_room_idx = None
                        st.rerun()
                    if c_cancel.form_submit_button("Cancel"):
                        st.session_state.edit_room_idx = None
                        st.rerun()
 
    # ── Teachers ──────────────────────────────────────────────────────────────
    with col2:
        st.subheader("👩‍🏫 Teachers")
        with st.form("teacher_form", clear_on_submit=True):
            t_name    = st.text_input("Teacher name")
            t_morning = st.checkbox("Prefers morning classes?")
            if st.form_submit_button("➕ Add Teacher") and t_name.strip():
                existing_names = {t.name for t in st.session_state.instructors}
                if t_name.strip() in existing_names:
                    st.error("A teacher with that name already exists.")
                else:
                    t_id = f"T{len(st.session_state.instructors)+1}"
                    st.session_state.instructors.append(
                        Instructor(t_id, t_name.strip(), t_morning)
                    )
                    st.success(f"Added teacher: {t_name}")
 
        if st.session_state.instructors:
            st.write("**Teachers:**")
            for i, t in enumerate(st.session_state.instructors):
                c_left, c_edit, c_del = st.columns([5, 1, 1])
                c_left.write(f"• **{t.name}**{' ☀️' if t.prefers_morning else ''}")
                if c_edit.button("✏️", key=f"edit_teacher_{i}"):
                    st.session_state.edit_teacher_idx = i
                if c_del.button("🗑️", key=f"del_teacher_{i}"):
                    st.session_state.instructors.pop(i)
                    if st.session_state.edit_teacher_idx == i:
                        st.session_state.edit_teacher_idx = None
                    st.rerun()
 
            ei = st.session_state.edit_teacher_idx
            if ei is not None and ei < len(st.session_state.instructors):
                et = st.session_state.instructors[ei]
                st.markdown("**Edit Teacher:**")
                with st.form("edit_teacher_form"):
                    new_name    = st.text_input("Name", value=et.name)
                    new_morning = st.checkbox("Prefers morning?", value=et.prefers_morning)
                    c_save, c_cancel = st.columns(2)
                    if c_save.form_submit_button("💾 Save"):
                        st.session_state.instructors[ei].name            = new_name.strip()
                        st.session_state.instructors[ei].prefers_morning = new_morning
                        st.session_state.edit_teacher_idx = None
                        st.rerun()
                    if c_cancel.form_submit_button("Cancel"):
                        st.session_state.edit_teacher_idx = None
                        st.rerun()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Courses & Divisions with Add / Edit / Remove
# ══════════════════════════════════════════════════════════════════════════════
with tab_courses:
 
    # ── Courses ───────────────────────────────────────────────────────────────
    st.subheader("📚 Courses")
    st.info("**Types:** Lecture (single division), Lab (batches), Minor (cross-division).")
    if not st.session_state.instructors:
        st.warning("Add at least one teacher first.")
    else:
        with st.form("course_form", clear_on_submit=True):
            c_name       = st.text_input("Course name")
            c_type       = st.radio("Course type", ["Lecture", "Lab", "Minor"], horizontal=True)
            c_students   = st.number_input("Max students", 5, value=60)
            c_weeks      = st.number_input("Classes per week", 1, 10, 3)
            sel_teachers = st.multiselect(
                "Assign teachers", [t.name for t in st.session_state.instructors]
            )
            enrolled_div_names = []
            if c_type == "Minor" and st.session_state.sections:
                enrolled_div_names = st.multiselect(
                    "Enrolled divisions (besides host)",
                    [s.name for s in st.session_state.sections],
                )
            if st.form_submit_button("➕ Add Course") and c_name.strip() and sel_teachers:
                assigned     = [t for t in st.session_state.instructors if t.name in sel_teachers]
                c_id         = f"C{len(st.session_state.courses)+1}"
                enrolled_ids = [s.id for s in st.session_state.sections
                                if s.name in enrolled_div_names]
                st.session_state.courses.append(
                    Course(c_id, c_name.strip(), c_students, assigned, c_weeks,
                           c_type, enrolled_ids)
                )
                st.success(f"Added {c_type} course: {c_name}")
 
        if st.session_state.courses:
            st.write("**Courses:**")
            for i, c in enumerate(st.session_state.courses):
                teachers = ", ".join(t.name for t in c.instructors)
                enrolled_names = [s.name for s in st.session_state.sections
                                  if s.id in c.enrolled_sections]
                extra = (f" | enrolled: {', '.join(enrolled_names)}"
                         if c.course_type == "Minor" and enrolled_names else "")
                label = (f"**{c.name}** ({c.course_type}) — "
                         f"{c.classes_per_week}/wk — teachers: {teachers}{extra}")
                c_left, c_edit, c_del = st.columns([6, 1, 1])
                c_left.markdown(f"• {label}")
                if c_edit.button("✏️", key=f"edit_course_{i}"):
                    st.session_state.edit_course_idx = i
                if c_del.button("🗑️", key=f"del_course_{i}"):
                    removed = st.session_state.courses.pop(i)
                    # Remove from all section_courses
                    for s_id in list(st.session_state.section_courses):
                        st.session_state.section_courses[s_id] = [
                            x for x in st.session_state.section_courses[s_id]
                            if x.id != removed.id
                        ]
                    if st.session_state.edit_course_idx == i:
                        st.session_state.edit_course_idx = None
                    st.rerun()
 
            ei = st.session_state.edit_course_idx
            if ei is not None and ei < len(st.session_state.courses):
                ec = st.session_state.courses[ei]
                st.markdown("**Edit Course:**")
                with st.form("edit_course_form"):
                    new_c_name  = st.text_input("Name", value=ec.name)
                    new_c_weeks = st.number_input("Classes/week", 1, 10,
                                                  int(ec.classes_per_week))
                    new_c_students = st.number_input("Max students", 5, 500,
                                                     int(ec.max_students))
                    new_teachers = st.multiselect(
                        "Teachers",
                        [t.name for t in st.session_state.instructors],
                        default=[t.name for t in ec.instructors
                                 if t.name in [x.name for x in st.session_state.instructors]],
                    )
                    c_save, c_cancel = st.columns(2)
                    if c_save.form_submit_button("💾 Save"):
                        ec.name             = new_c_name.strip()
                        ec.classes_per_week = new_c_weeks
                        ec.max_students     = new_c_students
                        ec.instructors      = [t for t in st.session_state.instructors
                                               if t.name in new_teachers]
                        st.session_state.edit_course_idx = None
                        st.rerun()
                    if c_cancel.form_submit_button("Cancel"):
                        st.session_state.edit_course_idx = None
                        st.rerun()
 
    st.markdown("---")
 
    # ── Divisions ─────────────────────────────────────────────────────────────
    st.subheader("🏢 Divisions (Sections)")
    if not st.session_state.courses:
        st.warning("Add at least one course first.")
    else:
        with st.form("section_form", clear_on_submit=True):
            s_name      = st.text_input("Division name")
            s_students  = st.number_input("Total students", 1, value=60)
            s_batches   = st.number_input("Number of lab batches", 1, 10, 4)
            sel_courses = st.multiselect(
                "Select courses for this division",
                [c.name for c in st.session_state.courses],
            )
            if st.form_submit_button("➕ Add Division") and s_name.strip() and sel_courses:
                s_id = f"S{len(st.session_state.sections)+1}"
                st.session_state.sections.append(
                    DepartmentSection(s_id, s_name.strip(), s_students, s_batches)
                )
                st.session_state.section_courses[s_id] = [
                    c for c in st.session_state.courses if c.name in sel_courses
                ]
                st.success(f"Added division: {s_name} with {s_batches} batches.")
 
        if st.session_state.sections:
            st.write("**Divisions:**")
            for i, s in enumerate(st.session_state.sections):
                courses_for_s = st.session_state.section_courses.get(s.id, [])
                label = (f"**{s.name}** — {s.number_of_students} students, "
                         f"{len(s.batches)} batches — "
                         f"courses: {', '.join(c.name for c in courses_for_s)}")
                c_left, c_edit, c_del = st.columns([6, 1, 1])
                c_left.markdown(f"• {label}")
                if c_edit.button("✏️", key=f"edit_sec_{i}"):
                    st.session_state.edit_section_idx = i
                if c_del.button("🗑️", key=f"del_sec_{i}"):
                    removed_s = st.session_state.sections.pop(i)
                    st.session_state.section_courses.pop(removed_s.id, None)
                    if st.session_state.edit_section_idx == i:
                        st.session_state.edit_section_idx = None
                    st.rerun()
 
            ei = st.session_state.edit_section_idx
            if ei is not None and ei < len(st.session_state.sections):
                es = st.session_state.sections[ei]
                current_courses = st.session_state.section_courses.get(es.id, [])
                st.markdown("**Edit Division:**")
                with st.form("edit_section_form"):
                    new_s_name     = st.text_input("Name", value=es.name)
                    new_s_students = st.number_input("Total students", 1, 2000,
                                                     int(es.number_of_students))
                    new_s_batches  = st.number_input("Lab batches", 1, 10,
                                                     int(len(es.batches)))
                    new_s_courses  = st.multiselect(
                        "Courses",
                        [c.name for c in st.session_state.courses],
                        default=[c.name for c in current_courses
                                 if c.name in [x.name for x in st.session_state.courses]],
                    )
                    c_save, c_cancel = st.columns(2)
                    if c_save.form_submit_button("💾 Save"):
                        st.session_state.sections[ei].name               = new_s_name.strip()
                        st.session_state.sections[ei].number_of_students = new_s_students
                        # rebuild batches list to match new count
                        st.session_state.sections[ei].batches = [
                            f"B{b+1}" for b in range(new_s_batches)
                        ]
                        st.session_state.section_courses[es.id] = [
                            c for c in st.session_state.courses if c.name in new_s_courses
                        ]
                        st.session_state.edit_section_idx = None
                        st.rerun()
                    if c_cancel.form_submit_button("Cancel"):
                        st.session_state.edit_section_idx = None
                        st.rerun()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Generate Timetable
# FIX: GA was stuck on "Initialising…" because ga.initialize_population() can
# throw silently, or evolve() can stall if the data object is built incorrectly.
# We now wrap the entire GA run in a try/except and surface any error clearly.
# The progress bar is updated every generation so Streamlit re-renders correctly.
# ══════════════════════════════════════════════════════════════════════════════
with tab_generate:
    st.subheader("Generate Master Timetable")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rooms",     len(st.session_state.rooms))
    c2.metric("Teachers",  len(st.session_state.instructors))
    c3.metric("Courses",   len(st.session_state.courses))
    c4.metric("Divisions", len(st.session_state.sections))
 
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
                if sev == "ERROR":
                    st.error(f"❌ ERROR: {msg}")
                elif sev == "WARNING":
                    st.warning(f"⚠️ WARNING: {msg}")
                else:
                    st.info(f"ℹ️ {msg}")
 
    st.markdown("---")
    st.markdown("### Step 2 — Generate")
    if st.button("🚀 Generate Master Timetable", type="primary"):
        errors = []
        if not st.session_state.rooms:    errors.append("No rooms added.")
        if not st.session_state.courses:  errors.append("No courses added.")
        if not st.session_state.sections: errors.append("No divisions added.")
        if not any(r.room_type == "Lecture" for r in st.session_state.rooms):
            errors.append("No Lecture rooms.")
        if (any(c.course_type == "Lab" for c in st.session_state.courses)
                and not any(r.room_type == "Lab" for r in st.session_state.rooms)):
            errors.append("Lab courses exist but no Lab rooms.")
        if errors:
            for e in errors:
                st.error(e)
            st.stop()
 
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
                tournament_size=tournament_sz,
                stagnation_limit=stagnation_lim,
            )
 
            progress_bar  = st.progress(0, text="Initialising…")
            status_text   = st.empty()
            log_placeholder = st.empty()
 
            # ── initialise population (the step that was silently failing) ──
            try:
                population = ga.initialize_population()
            except Exception as init_err:
                st.error(f"❌ Failed to initialise population: {init_err}")
                st.exception(init_err)
                st.stop()
 
            if not population:
                st.error("Population is empty after initialisation — check Data/models setup.")
                st.stop()
 
            run_log       = []
            best_schedule = None
            soft_refine   = 0
            SOFT_REFINE_GEN = 30
 
            for gen in range(int(generations)):
                try:
                    population = ga.evolve(population)
                except Exception as evolve_err:
                    st.error(f"❌ Error at generation {gen+1}: {evolve_err}")
                    st.exception(evolve_err)
                    break
 
                if not population:
                    st.error(f"Population became empty at generation {gen+1}.")
                    break
 
                best_schedule = population[0]
                pct = (gen + 1) / int(generations)
                progress_bar.progress(
                    pct,
                    text=f"Generation {gen+1}/{int(generations)} | "
                         f"fitness={best_schedule.fitness:.4f} | "
                         f"conflicts={best_schedule.hard_conflicts}",
                )
                log_entry = (
                    f"Gen {gen+1:4d} | fitness={best_schedule.fitness:.6f} | "
                    f"hard={best_schedule.hard_conflicts} | mut_rate={ga.mutation_rate:.3f}"
                )
                run_log.append(log_entry)
                status_text.text(log_entry)
 
                if best_schedule.hard_conflicts == 0:
                    soft_refine += 1
                    if soft_refine >= SOFT_REFINE_GEN:
                        progress_bar.progress(
                            1.0, text=f"Early stop — {gen+1} generations (converged)"
                        )
                        break
                else:
                    soft_refine = 0
 
            if best_schedule is not None:
                progress_bar.progress(1.0, text="Done!")
                st.session_state.best_schedule = best_schedule
                st.session_state.run_log       = run_log
                log_placeholder.text_area(
                    "Run log (last 30)", "\n".join(run_log[-30:]), height=150
                )
            else:
                st.error("No schedule produced — check your data and try again.")
 
        except Exception as outer_err:
            st.error(f"❌ Unexpected error: {outer_err}")
            st.exception(outer_err)
 
    # ── Display result ────────────────────────────────────────────────────────
    best = st.session_state.best_schedule
    if best is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Final Fitness",  f"{best.fitness:.6f}")
        col_b.metric("Hard Conflicts", best.hard_conflicts,
                     delta=None if best.hard_conflicts > 0 else "✅ Zero")
        col_c.metric("Total Classes",  len(best.classes))
        if best.hard_conflicts > 0:
            st.warning(
                f"Schedule has {best.hard_conflicts} conflict(s). "
                "Try increasing population/generations."
            )
        else:
            st.success("Perfect schedule – zero hard conflicts!")
 
        st.subheader("Master Timetable Matrix")
        df_pivot = _schedule_to_df(best)
        st.dataframe(df_pivot, use_container_width=True, height=800)
        csv_bytes = df_pivot.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv_bytes,
                           "master_timetable.csv", "text/csv")
 
        st.subheader("Per-Division Timetable")
        div_names = sorted({c.section.name for c in best.classes})
        if div_names:
            selected_div = st.selectbox("Select division", div_names)
            div_rows = []
            for c in best.classes:
                if c.section.name != selected_div:
                    continue
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
                    "_sort": {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5}[
                        c.meeting_time.day
                    ],
                })
            if div_rows:
                df_div = (pd.DataFrame(div_rows)
                          .sort_values(["_sort", "Time"])
                          .drop(columns=["_sort"]))
                st.dataframe(df_div, use_container_width=True, height=500)
 
        minor_events = [c for c in best.classes if c.course.course_type == "Minor"]
        if minor_events:
            st.subheader("📗 Minor Course Schedule")
            minor_rows = []
            for c in minor_events:
                aff_names = [
                    s.name for s in st.session_state.sections
                    if s.id in c.get_affected_section_ids()
                ]
                minor_rows.append({
                    "Course":   c.course.name,
                    "Day":      c.meeting_time.day,
                    "Time":     c.meeting_time.time_str,
                    "Teacher":  c.instructor.name,
                    "Room":     c.room.id,
                    "Sections": ", ".join(sorted(aff_names)),
                })
            st.dataframe(pd.DataFrame(minor_rows), use_container_width=True)
 
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