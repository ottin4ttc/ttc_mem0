import OpenAI from "openai";
import { Embedder } from "./base";
import { EmbeddingConfig } from "../types";

export class OpenAIEmbedder implements Embedder {
  private openai: OpenAI;
  private model: string;
  private dimension: number;

  constructor(config: EmbeddingConfig) {
    this.openai = new OpenAI({ apiKey: config.apiKey, baseURL: config.url });
    this.model = config.model || "text-embedding-3-small";
    this.dimension = config.dimension || 1536;
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.openai.embeddings.create({
      model: this.model,
      input: text,
      dimensions: this.dimension,
    });
    return response.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await this.openai.embeddings.create({
      model: this.model,
      input: texts,
      dimensions: this.dimension,
    });
    return response.data.map((item) => item.embedding);
  }
}
