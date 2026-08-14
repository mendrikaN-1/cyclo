-- =====================================================================
-- Table pour stocker les dates de règles et la durée réelle de chaque cycle
-- À exécuter dans Supabase : SQL Editor > New query > coller > Run
-- =====================================================================

create table if not exists cycles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  date_debut_regles date not null,
  duree_regles int default 5 check (duree_regles >= 1 and duree_regles <= 14),
  created_at timestamp with time zone default now()
);

-- Index pour accélérer les requêtes
create index if not exists cycles_user_id_idx on cycles(user_id);

-- Activer Row Level Security (RLS)
alter table cycles enable row level security;

-- Politiques de sécurité
drop policy if exists "Les utilisatrices peuvent lire leurs propres cycles" on cycles;
create policy "Les utilisatrices peuvent lire leurs propres cycles"
  on cycles for select
  using (auth.uid() = user_id);

drop policy if exists "Les utilisatrices peuvent ajouter leurs propres cycles" on cycles;
create policy "Les utilisatrices peuvent ajouter leurs propres cycles"
  on cycles for insert
  with check (auth.uid() = user_id);

drop policy if exists "Les utilisatrices peuvent modifier leurs propres cycles" on cycles;
create policy "Les utilisatrices peuvent modifier leurs propres cycles"
  on cycles for update
  using (auth.uid() = user_id);

drop policy if exists "Les utilisatrices peuvent supprimer leurs propres cycles" on cycles;
create policy "Les utilisatrices peuvent supprimer leurs propres cycles"
  on cycles for delete
  using (auth.uid() = user_id);
