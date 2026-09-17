import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn(
    'Faltan configurar las variables de entorno de Supabase: VITE_SUPABASE_URL y/o VITE_SUPABASE_ANON_KEY en tu archivo .env.local'
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

