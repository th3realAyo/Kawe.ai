create table users (
    id uuid primary key default gen_random_uuid(),
    username_encrypted text not null,
    created_at timestamptz not null default now()
);

create table usage_events (
    id bigserial primary key,
    user_id uuid not null references users(id) on delete cascade,
    document_id uuid not null default gen_random_uuid(),
    uploaded_at timestamptz not null default now(),
    document_type text,
    document_length_words integer,
    summarization_percentage integer,
    processing_success boolean,
    error_type text,
    summary_language text,
    audio_generated boolean default false,
    audio_language text,
    voice_selected text,
    session_duration_seconds integer,
    time_of_day text,
    day_of_week text
);

create index on usage_events (user_id);
create index on usage_events (uploaded_at);

-- `usage_events.id` is backed by a PostgreSQL sequence. Inserting a row
-- calls nextval() on that sequence, which requires USAGE independently of
-- the table INSERT grant. Kàwé writes usage events only from its server-side
-- Supabase client, so grant the minimum required permission to service_role.
grant usage on sequence public.usage_events_id_seq to service_role;
