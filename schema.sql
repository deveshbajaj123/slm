-- LMB History & Civics question bank storage schema.
-- Regenerated from scratch by build_index.py on every run.

CREATE TABLE context_groups (
    id                  TEXT PRIMARY KEY,
    context_type        TEXT,
    context_description TEXT,
    topic               TEXT,
    total_marks         INTEGER,
    class               TEXT,
    year                TEXT,
    exam_type           TEXT,
    school              TEXT
);

CREATE TABLE questions (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,            -- 'standalone' | 'context_sub'
    type             TEXT,                     -- NULL for context_sub
    marks            INTEGER NOT NULL,
    question_text    TEXT NOT NULL,
    topic            TEXT,
    subtopic         TEXT,
    difficulty       TEXT,
    bloom_level      TEXT,
    class            TEXT,
    year             TEXT,
    exam_type        TEXT,
    school           TEXT,
    context_group_id TEXT REFERENCES context_groups(id),
    sub_label        TEXT,
    text_hash        TEXT NOT NULL
);

CREATE INDEX idx_questions_type_marks   ON questions(type, marks);
CREATE INDEX idx_questions_class_year   ON questions(class, year);
CREATE INDEX idx_questions_topic        ON questions(topic);
CREATE INDEX idx_questions_difficulty   ON questions(difficulty);
CREATE INDEX idx_questions_bloom        ON questions(bloom_level);
CREATE INDEX idx_questions_hash         ON questions(text_hash);
CREATE INDEX idx_questions_cg           ON questions(context_group_id);
