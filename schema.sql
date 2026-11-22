-- Supabase tables used by kawe/db.py. Run once in the Supabase SQL editor.
--
-- Email addresses are never stored: users.email_hash is an HMAC-SHA256 of the lower-cased
-- address under the app's EMAIL_HASH_KEY secret, and cannot be reversed.

create table if not exists users (
    id uuid primary key default gen_random_uuid(),
    email_hash text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists usage_events (
    id bigint generated always as identity primary key,
    user_id uuid not null references users (id) on delete cascade,
    document_type text,
    document_length_words integer,
    summarization_percentage integer,
    processing_success boolean,
    error_type text,  -- a short category such as 'summarization' or 'tts', never error text
    summary_language text,
    audio_generated boolean default false,
    audio_language text,
    voice_selected text,
    event_type text not null default 'summary',  -- 'summary' or 'question'; question text is never stored
    question_answered boolean,                  -- for questions: whether the document answered it
    time_of_day text,
    day_of_week text,
    created_at timestamptz not null default now()
);

-- The app connects with the service key, which bypasses row level security. Enabling RLS with
-- no policies means the public (anon) key can read or write nothing in these tables.
alter table users enable row level security;
alter table usage_events enable row level security;

-- Migrating a database created by the earlier username sign-up (users.username_encrypted):
-- existing accounts have no email, so they cannot sign in again. Either delete them
--     delete from users;            -- also removes their usage_events
-- or keep their statistics by giving each a placeholder hash before tightening the column:
--     alter table users add column if not exists email_hash text;
--     update users set email_hash = 'legacy-' || id where email_hash is null;
--     alter table users alter column email_hash set not null;
--     alter table users add constraint users_email_hash_key unique (email_hash);
--     alter table users drop column username_encrypted;
-- Then run the two "enable row level security" lines above.

-- Adding the ask-the-document prototype to an existing database:
--     alter table usage_events add column if not exists event_type text not null default 'summary';
--     alter table usage_events add column if not exists question_answered boolean;
