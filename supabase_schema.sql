-- =====================================================================
-- Cyclo — Schéma complet de la table `cycles`
-- À exécuter dans Supabase : SQL Editor > New query > coller > Run
--
-- Ce script est sûr à exécuter plusieurs fois (idempotent) : il ne
-- provoquera pas d'erreur si la table ou les colonnes existent déjà.
-- =====================================================================

-- 1. Création de la table (si elle n'existe pas déjà)
create table if not exists cycles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  date_debut_regles date not null,
  duree_regles int default 5 check (duree_regles >= 1 and duree_regles <= 14),
  created_at timestamp with time zone default now()
);

-- 2. Colonne pour la vraie date de fin des règles (v4 : calcul réel de la
--    durée à partir de deux dates, au lieu d'un chiffre saisi manuellement)
alter table cycles add column if not exists date_fin_regles date;

-- 3. Index pour accélérer les requêtes "donne-moi les cycles de tel utilisateur"
create index if not exists cycles_user_id_idx on cycles(user_id);

-- =====================================================================
-- Row Level Security (RLS) : chaque utilisatrice ne peut voir/modifier
-- QUE ses propres données, jamais celles des autres. Indispensable pour
-- la confidentialité de données aussi sensibles.
-- =====================================================================

alter table cycles enable row level security;

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
