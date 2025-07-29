import { SearchFilters } from "../types";

export interface Entity {
  userId?: string;
  agentId?: string;
  runId?: string;
}

export interface Tag {
  name: string;
  description: string;
}

export interface AddMemoryOptions extends Entity {
  metadata?: Record<string, any>;
  filters?: SearchFilters;
  infer?: boolean;
  tags?: Tag[];
  customPrompt?: string;
}

export interface SearchMemoryOptions extends Entity {
  limit?: number;
  filters?: SearchFilters;
}

export interface GetAllMemoryOptions extends Entity {
  limit?: number;
  filters?: SearchFilters;
}

export interface DeleteAllMemoryOptions extends Entity {}
