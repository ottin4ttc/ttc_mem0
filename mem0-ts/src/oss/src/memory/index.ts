import { v4 as uuidv4 } from "uuid";
import { createHash } from "crypto";
import {
  AddToVectorStoreOptions,
  MemoryAction,
  MemoryConfig,
  MemoryConfigSchema,
  MemoryItem,
  Message,
  SearchFilters,
  SearchResult,
  VectorStoreResult,
} from "../types";
import {
  EmbedderFactory,
  LLMFactory,
  VectorStoreFactory,
  HistoryManagerFactory,
} from "../utils/factory";
import {
  getFactRetrievalMessages,
  getTextTagRetrievalMessages,
  getTextTagUpdateMemoryMessages,
  getUpdateMemoryMessages,
  removeCodeBlocks,
} from "../prompts";
import { DummyHistoryManager } from "../storage/DummyHistoryManager";
import { Embedder } from "../embeddings/base";
import { LLM } from "../llms/base";
import { VectorStore } from "../vector_stores/base";
import { ConfigManager } from "../config/manager";
import { MemoryGraph } from "./graph_memory";
import {
  AddMemoryOptions,
  SearchMemoryOptions,
  DeleteAllMemoryOptions,
  GetAllMemoryOptions,
  Tag,
} from "./memory.types";
import { parse_vision_messages } from "../utils/memory";
import { HistoryManager } from "../storage/base";
import { captureClientEvent } from "../utils/telemetry";
import { TagsUtils } from "../utils/tags";

export class Memory {
  private config: MemoryConfig;
  private customPrompt: string | undefined;
  private embedder: Embedder;
  private vectorStore: VectorStore;
  private llm: LLM;
  private db: HistoryManager;
  private collectionName: string | undefined;
  private apiVersion: string;
  private graphMemory?: MemoryGraph;
  private enableGraph: boolean;
  telemetryId: string;

  constructor(config: Partial<MemoryConfig> = {}) {
    // Merge and validate config
    this.config = ConfigManager.mergeConfig(config);

    this.customPrompt = this.config.customPrompt;
    this.embedder = EmbedderFactory.create(
      this.config.embedder.provider,
      this.config.embedder.config,
    );
    this.vectorStore = VectorStoreFactory.create(
      this.config.vectorStore.provider,
      this.config.vectorStore.config,
    );
    this.llm = LLMFactory.create(
      this.config.llm.provider,
      this.config.llm.config,
    );
    if (this.config.disableHistory) {
      this.db = new DummyHistoryManager();
    } else {
      const defaultConfig = {
        provider: "sqlite",
        config: {
          historyDbPath: this.config.historyDbPath || ":memory:",
        },
      };

      this.db =
        this.config.historyStore && !this.config.disableHistory
          ? HistoryManagerFactory.create(
              this.config.historyStore.provider,
              this.config.historyStore,
            )
          : HistoryManagerFactory.create("sqlite", defaultConfig);
    }

    this.collectionName = this.config.vectorStore.config.collectionName;
    this.apiVersion = this.config.version || "v1.0";
    this.enableGraph = this.config.enableGraph || false;
    this.telemetryId = "anonymous";

    // Initialize graph memory if configured
    if (this.enableGraph && this.config.graphStore) {
      this.graphMemory = new MemoryGraph(this.config);
    }

    // Initialize telemetry if vector store is initialized
    this._initializeTelemetry();
  }

  private async _initializeTelemetry() {
    try {
      await this._getTelemetryId();

      // Capture initialization event
      await captureClientEvent("init", this, {
        api_version: this.apiVersion,
        client_type: "Memory",
        collection_name: this.collectionName,
        enable_graph: this.enableGraph,
      });
    } catch (error) {}
  }

  private async _getTelemetryId() {
    try {
      if (
        !this.telemetryId ||
        this.telemetryId === "anonymous" ||
        this.telemetryId === "anonymous-supabase"
      ) {
        this.telemetryId = await this.vectorStore.getUserId();
      }
      return this.telemetryId;
    } catch (error) {
      this.telemetryId = "anonymous";
      return this.telemetryId;
    }
  }

  private async _captureEvent(methodName: string, additionalData = {}) {
    try {
      await this._getTelemetryId();
      await captureClientEvent(methodName, this, {
        ...additionalData,
        api_version: this.apiVersion,
        collection_name: this.collectionName,
      });
    } catch (error) {
      console.error(`Failed to capture ${methodName} event:`, error);
    }
  }

  static fromConfig(configDict: Record<string, any>): Memory {
    try {
      const config = MemoryConfigSchema.parse(configDict);
      return new Memory(config);
    } catch (e) {
      console.error("Configuration validation error:", e);
      throw e;
    }
  }

  async add(
    messages: string | Message[],
    config: AddMemoryOptions,
  ): Promise<SearchResult> {
    await this._captureEvent("add", {
      message_count: Array.isArray(messages) ? messages.length : 1,
      has_metadata: !!config.metadata,
      has_filters: !!config.filters,
      infer: config.infer,
    });
    const {
      userId,
      agentId,
      runId,
      metadata = {},
      filters = {},
      infer = true,
      tags = [],
      customPrompt = "",
    } = config;

    if (userId) filters.userId = metadata.userId = userId;
    if (agentId) filters.agentId = metadata.agentId = agentId;
    if (runId) filters.runId = metadata.runId = runId;

    if (!filters.userId && !filters.agentId && !filters.runId) {
      throw new Error(
        "One of the filters: userId, agentId or runId is required!",
      );
    }

    const parsedMessages = Array.isArray(messages)
      ? (messages as Message[])
      : [{ role: "user", content: messages }];

    const final_parsedMessages = await parse_vision_messages(parsedMessages);

    // Add to vector store
    let vectorStoreResult: MemoryItem[] = [];
    if (!infer) {
      vectorStoreResult = await this.addToVectorStoreDirect({
        messages: final_parsedMessages,
        tags,
        metadata,
      });
    } else {
      vectorStoreResult = await this.addToVectorStore(
        final_parsedMessages,
        tags,
        customPrompt,
        metadata,
        filters,
      );
    }

    // Add to graph store if available
    let graphResult;
    if (this.graphMemory) {
      try {
        graphResult = await this.graphMemory.add(
          final_parsedMessages.map((m) => m.content).join("\n"),
          filters,
        );
      } catch (error) {
        console.error("Error adding to graph memory:", error);
      }
    }

    return {
      results: vectorStoreResult,
      relations: graphResult?.relations,
    };
  }

  async createActions(
    messages: string | Message[],
    config: AddMemoryOptions,
  ): Promise<MemoryAction[]> {
    await this._captureEvent("create_actions", {
      message_count: Array.isArray(messages) ? messages.length : 1,
      has_metadata: !!config.metadata,
      has_filters: !!config.filters,
      infer: config.infer,
    });
    const {
      userId,
      agentId,
      runId,
      tags = [],
      metadata = {},
      filters = {},
    } = config;

    if (userId) filters.userId = metadata.userId = userId;
    if (agentId) filters.agentId = metadata.agentId = agentId;
    if (runId) filters.runId = metadata.runId = runId;

    if (!filters.userId && !filters.agentId && !filters.runId) {
      throw new Error(
        "One of the filters: userId, agentId or runId is required!",
      );
    }

    const parsedMessages = Array.isArray(messages)
      ? (messages as Message[])
      : [{ role: "user", content: messages }];

    const final_parsedMessages = await parse_vision_messages(parsedMessages);

    // Add to vector store
    const actions = await this.createAddToVectorStoreActions({
      messages: final_parsedMessages,
      tags,
      filters,
      customPrompt: config.customPrompt,
      metadata,
    });

    return actions;
  }

  async applyActions(actions: MemoryAction[]): Promise<MemoryItem[]> {
    const results: MemoryItem[] = [];
    for (const action of actions) {
      try {
        const tagNames = action.tagNames || [];
        const data = TagsUtils.wrapData(action.text, tagNames);
        switch (action.event) {
          case "ADD": {
            const memoryId = await this.createMemory(
              data,
              tagNames,
              {},
              action.metadata || {},
            );
            results.push({
              id: memoryId,
              memory: data,
              metadata: { event: "ADD" },
            });
            break;
          }
          case "UPDATE": {
            const memoryId = await this.updateMemory(
              action.memoryId,
              data,
              tagNames,
              {},
              action.metadata || {},
            );
            results.push({
              id: memoryId,
              memory: data,
              metadata: { event: "UPDATE" },
            });
            break;
          }
          case "DELETE": {
            await this.deleteMemory(action.memoryId);
            results.push({
              id: action.memoryId,
              memory: "",
              metadata: { event: "DELETE" },
            });
            break;
          }
        }
      } catch (error) {
        console.error(`Error apply actions: ${error}`);
      }
    }
    return results;
  }

  private async createAddToVectorStoreActions(
    options: AddToVectorStoreOptions,
  ): Promise<MemoryAction[]> {
    const { messages, tags, filters, customPrompt, metadata } = options;
    const parsedMessages = messages.map((m) => m.content).join("\n");

    // 设置 system prompt 和 user prompt
    let systemPrompt = "";
    let userPrompt = "";

    if (customPrompt || this.customPrompt) {
      systemPrompt = customPrompt || this.customPrompt || "";
      userPrompt =
        "Following is a conversation between the user and the assistant." +
        "You have to extract the relevant facts and tags about the user, if any, from the conversation and return them in the JSON format as shown above.";
      if (tags.length > 0) {
        userPrompt += `\n\n### Input\n${parsedMessages}\n\n### Tags\n${tags.map((tag, index) => `${index + 1}. ${tag.name}: ${tag.description}`).join("\n")}`;
      } else {
        userPrompt += `\n\n### Input\n${parsedMessages}`;
      }
    } else if (tags.length > 0) {
      [systemPrompt, userPrompt] = getTextTagRetrievalMessages(
        parsedMessages,
        tags,
      );
    } else {
      [systemPrompt, userPrompt] = getFactRetrievalMessages(parsedMessages);
    }

    // 调用 LLM 获取 facts 和 tagNamesList
    const response = await this.llm.generateResponse(
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      { type: "json_object" },
    );
    const cleanResponse = removeCodeBlocks(response as string);
    let facts: string[] = [];
    let tagNamesList: string[][] = [];
    try {
      const responseJson = JSON.parse(cleanResponse);
      facts = responseJson.facts || [];
      tagNamesList = responseJson.tagNamesList || [];
    } catch (e) {
      console.error(
        "Failed to parse facts from LLM response:",
        cleanResponse,
        e,
      );
      facts = [];
      tagNamesList = [];
    }

    // Get embeddings for new facts
    const newMessageEmbeddings: Record<string, number[]> = {};
    const retrievedOldMemory: Array<{
      id: string;
      text: string;
      tag_names: string[];
    }> = [];
    const factsWithTagNames: Array<{
      text: string;
      tag_names: string[];
    }> = [];

    // Create embeddings and search for similar memories
    for (let i = 0; i < facts.length; i++) {
      const tagNames: string[] = tagNamesList[i] || [];
      const fact =
        Array.isArray(tagNames) && tagNames.length > 0
          ? ` ${TagsUtils.getDataPrefix(tagNames)} ${facts[i]}`
          : facts[i];
      factsWithTagNames.push({ text: fact, tag_names: tagNames });
      facts[i] = fact;

      const embedding = await this.embedder.embed(fact);
      newMessageEmbeddings[fact] = embedding;

      const existingMemories = await this.vectorStore.search(
        embedding,
        5,
        filters,
      );
      for (const mem of existingMemories) {
        retrievedOldMemory.push({
          id: mem.id,
          text: mem.payload.data,
          tag_names: mem.payload.tagNames,
        });
      }
    }

    // Remove duplicates from old memories
    const uniqueOldMemories = retrievedOldMemory.filter(
      (mem, index) =>
        retrievedOldMemory.findIndex((m) => m.id === mem.id) === index,
    );
    // Create UUID mapping for handling UUID hallucinations
    const tempUuidMapping: Record<string, string> = {};
    uniqueOldMemories.forEach((item, idx) => {
      tempUuidMapping[String(idx)] = item.id;
      uniqueOldMemories[idx].id = String(idx);
    });

    // Get memory update decisions
    let updatePrompt = "";
    if (tags.length > 0) {
      updatePrompt = getTextTagUpdateMemoryMessages(
        uniqueOldMemories,
        factsWithTagNames,
      );
    } else {
      updatePrompt = getUpdateMemoryMessages(uniqueOldMemories, facts);
    }

    const updateResponse = await this.llm.generateResponse(
      [{ role: "user", content: updatePrompt }],
      { type: "json_object" },
    );

    const cleanUpdateResponse = removeCodeBlocks(updateResponse as string);
    let actions: {
      id: string;
      event: "ADD" | "UPDATE" | "DELETE" | "NONE";
      text: string;
      tag_names?: string[];
      old_memory?: string;
      old_tag_names?: string[];
    }[] = [];

    try {
      actions = JSON.parse(cleanUpdateResponse).memory || [];
    } catch (e) {
      console.error(
        "Failed to parse memory actions from LLM response:",
        cleanUpdateResponse,
        e,
      );
      actions = [];
    }

    const memoryActions: MemoryAction[] = [];
    actions.forEach((action) => {
      if (action.event === "NONE") {
        return;
      }
      const tagNames = action.tag_names || [];
      const oldTagNames = action.old_tag_names || [];
      memoryActions.push({
        actionId: uuidv4(),
        memoryId: tempUuidMapping[action.id],
        event: action.event,
        text: TagsUtils.clarifyData(action.text, tagNames),
        tagNames,
        oldText: TagsUtils.clarifyData(action.old_memory || "", oldTagNames),
        oldTagNames,
        metadata,
      });
    });

    return memoryActions;
  }

  private async addToVectorStoreDirect(options: {
    messages: Message[];
    tags: Tag[];
    metadata: Record<string, any>;
  }): Promise<MemoryItem[]> {
    const { messages, metadata } = options;
    const returnedMemories: MemoryItem[] = [];
    for (const message of messages) {
      if (message.content === "system") {
        continue;
      }
      const memoryId = await this.createMemory(
        message.content as string,
        options.tags.map((tag) => tag.name),
        {},
        metadata,
      );
      returnedMemories.push({
        id: memoryId,
        memory: message.content as string,
        metadata: { event: "ADD" },
      });
    }
    return returnedMemories;
  }

  private async addToVectorStore(
    messages: Message[],
    tags: Tag[],
    customPrompt: string,
    metadata: Record<string, any>,
    filters: SearchFilters,
  ): Promise<MemoryItem[]> {
    const memoryActions = await this.createAddToVectorStoreActions({
      messages,
      tags,
      filters,
      customPrompt,
      metadata,
    });

    // Process memory actions
    const results = await this.applyActions(memoryActions);

    return results;
  }

  async get(memoryId: string): Promise<MemoryItem | null> {
    const memory = await this.vectorStore.get(memoryId);
    if (!memory) return null;

    const filters = {
      ...(memory.payload.userId && { userId: memory.payload.userId }),
      ...(memory.payload.agentId && { agentId: memory.payload.agentId }),
      ...(memory.payload.runId && { runId: memory.payload.runId }),
    };

    const memoryItem: MemoryItem = {
      id: memory.id,
      memory: memory.payload.data,
      hash: memory.payload.hash,
      createdAt: memory.payload.createdAt,
      updatedAt: memory.payload.updatedAt,
      metadata: {},
    };

    // Add additional metadata
    const excludedKeys = new Set([
      "userId",
      "agentId",
      "runId",
      "hash",
      "data",
      "createdAt",
      "updatedAt",
    ]);
    for (const [key, value] of Object.entries(memory.payload)) {
      if (!excludedKeys.has(key)) {
        memoryItem.metadata![key] = value;
      }
    }

    return { ...memoryItem, ...filters };
  }

  async search(
    query: string,
    config: SearchMemoryOptions,
  ): Promise<SearchResult> {
    await this._captureEvent("search", {
      query_length: query.length,
      limit: config.limit,
      has_filters: !!config.filters,
    });
    const { userId, agentId, runId, limit = 100, filters = {} } = config;

    if (userId) filters.userId = userId;
    if (agentId) filters.agentId = agentId;
    if (runId) filters.runId = runId;

    if (!filters.userId && !filters.agentId && !filters.runId) {
      throw new Error(
        "One of the filters: userId, agentId or runId is required!",
      );
    }

    // Search vector store
    const queryEmbedding = await this.embedder.embed(query);
    const memories = await this.vectorStore.search(
      queryEmbedding,
      limit,
      filters,
    );

    // Search graph store if available
    let graphResults;
    if (this.graphMemory) {
      try {
        graphResults = await this.graphMemory.search(query, filters);
      } catch (error) {
        console.error("Error searching graph memory:", error);
      }
    }

    const excludedKeys = new Set([
      "userId",
      "agentId",
      "runId",
      "hash",
      "data",
      "createdAt",
      "updatedAt",
    ]);
    const results = memories.map((mem) => ({
      id: mem.id,
      memory: mem.payload.data,
      hash: mem.payload.hash,
      createdAt: mem.payload.createdAt,
      updatedAt: mem.payload.updatedAt,
      score: mem.score,
      metadata: Object.entries(mem.payload)
        .filter(([key]) => !excludedKeys.has(key))
        .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {}),
      ...(mem.payload.userId && { userId: mem.payload.userId }),
      ...(mem.payload.agentId && { agentId: mem.payload.agentId }),
      ...(mem.payload.runId && { runId: mem.payload.runId }),
    }));

    return {
      results,
      relations: graphResults,
    };
  }

  async update(
    memoryId: string,
    data: string,
    tagNames?: string[],
  ): Promise<{ message: string }> {
    await this._captureEvent("update", { memory_id: memoryId });
    const embedding = await this.embedder.embed(data);
    await this.updateMemory(memoryId, data, tagNames || [], {
      [data]: embedding,
    });
    return { message: "Memory updated successfully!" };
  }

  async delete(memoryId: string): Promise<{ message: string }> {
    await this._captureEvent("delete", { memory_id: memoryId });
    await this.deleteMemory(memoryId);
    return { message: "Memory deleted successfully!" };
  }

  async deleteAll(
    config: DeleteAllMemoryOptions,
  ): Promise<{ message: string }> {
    await this._captureEvent("delete_all", {
      has_user_id: !!config.userId,
      has_agent_id: !!config.agentId,
      has_run_id: !!config.runId,
    });
    const { userId, agentId, runId } = config;

    const filters: SearchFilters = {};
    if (userId) filters.userId = userId;
    if (agentId) filters.agentId = agentId;
    if (runId) filters.runId = runId;

    if (!Object.keys(filters).length) {
      throw new Error(
        "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method.",
      );
    }

    const [memories] = await this.vectorStore.list(filters);
    for (const memory of memories) {
      await this.deleteMemory(memory.id);
    }

    return { message: "Memories deleted successfully!" };
  }

  async history(memoryId: string): Promise<any[]> {
    return this.db.getHistory(memoryId);
  }

  async reset(): Promise<void> {
    await this._captureEvent("reset");
    await this.db.reset();

    // Check provider before attempting deleteCol
    if (this.config.vectorStore.provider.toLowerCase() !== "langchain") {
      try {
        await this.vectorStore.deleteCol();
      } catch (e) {
        console.error(
          `Failed to delete collection for provider '${this.config.vectorStore.provider}':`,
          e,
        );
        // Decide if you want to re-throw or just log
      }
    } else {
      console.warn(
        "Memory.reset(): Skipping vector store collection deletion as 'langchain' provider is used. Underlying Langchain vector store data is not cleared by this operation.",
      );
    }

    if (this.graphMemory) {
      await this.graphMemory.deleteAll({ userId: "default" }); // Assuming this is okay, or needs similar check?
    }

    // Re-initialize factories/clients based on the original config
    this.embedder = EmbedderFactory.create(
      this.config.embedder.provider,
      this.config.embedder.config,
    );
    // Re-create vector store instance - crucial for Langchain to reset wrapper state if needed
    this.vectorStore = VectorStoreFactory.create(
      this.config.vectorStore.provider,
      this.config.vectorStore.config, // This will pass the original client instance back
    );
    this.llm = LLMFactory.create(
      this.config.llm.provider,
      this.config.llm.config,
    );
    // Re-init DB if needed (though db.reset() likely handles its state)
    // Re-init Graph if needed

    // Re-initialize telemetry
    this._initializeTelemetry();
  }

  async find(filters: SearchFilters): Promise<SearchResult> {
    const [memories] = await this.vectorStore.list(filters);
    return this.vectorResult2SearchResult(memories);
  }

  async getAll(config: GetAllMemoryOptions): Promise<SearchResult> {
    await this._captureEvent("get_all", {
      limit: config.limit,
      has_user_id: !!config.userId,
      has_agent_id: !!config.agentId,
      has_run_id: !!config.runId,
    });
    const { userId, agentId, runId, limit = 100 } = config;

    const filters: SearchFilters = {};
    if (userId) filters.userId = userId;
    if (agentId) filters.agentId = agentId;
    if (runId) filters.runId = runId;

    const [memories] = await this.vectorStore.list(filters, limit);
    return this.vectorResult2SearchResult(memories);
  }

  private async createMemory(
    data: string,
    tagNames: string[],
    existingEmbeddings: Record<string, number[]>,
    metadata: Record<string, any>,
  ): Promise<string> {
    const memoryId = uuidv4();
    const embedding =
      existingEmbeddings[data] || (await this.embedder.embed(data));

    const memoryMetadata = {
      ...metadata,
      tagNames,
      data,
      hash: createHash("md5").update(data).digest("hex"),
      createdAt: new Date().toISOString(),
    };

    await this.vectorStore.insert([embedding], [memoryId], [memoryMetadata]);
    await this.db.addHistory(
      memoryId,
      null,
      data,
      "ADD",
      memoryMetadata.createdAt,
    );

    return memoryId;
  }

  private async updateMemory(
    memoryId: string,
    data: string,
    tagNames: string[],
    existingEmbeddings: Record<string, number[]>,
    metadata: Record<string, any> = {},
  ): Promise<string> {
    const existingMemory = await this.vectorStore.get(memoryId);
    if (!existingMemory) {
      throw new Error(`Memory with ID ${memoryId} not found`);
    }

    const prevValue = existingMemory.payload.data;
    const embedding =
      existingEmbeddings[data] || (await this.embedder.embed(data));

    const newMetadata = {
      ...metadata,
      data,
      tagNames,
      hash: createHash("md5").update(data).digest("hex"),
      createdAt: existingMemory.payload.createdAt,
      updatedAt: new Date().toISOString(),
      ...(existingMemory.payload.userId && {
        userId: existingMemory.payload.userId,
      }),
      ...(existingMemory.payload.agentId && {
        agentId: existingMemory.payload.agentId,
      }),
      ...(existingMemory.payload.runId && {
        runId: existingMemory.payload.runId,
      }),
    };

    await this.vectorStore.update(memoryId, embedding, newMetadata);
    await this.db.addHistory(
      memoryId,
      prevValue,
      data,
      "UPDATE",
      newMetadata.createdAt,
      newMetadata.updatedAt,
    );

    return memoryId;
  }

  private async deleteMemory(memoryId: string): Promise<string> {
    const existingMemory = await this.vectorStore.get(memoryId);
    if (!existingMemory) {
      throw new Error(`Memory with ID ${memoryId} not found`);
    }

    const prevValue = existingMemory.payload.data;
    await this.vectorStore.delete(memoryId);
    await this.db.addHistory(
      memoryId,
      prevValue,
      null,
      "DELETE",
      undefined,
      undefined,
      1,
    );

    return memoryId;
  }

  private vectorResult2SearchResult(
    memories: VectorStoreResult[],
  ): SearchResult {
    const excludedKeys = new Set([
      "userId",
      "agentId",
      "runId",
      "hash",
      "data",
      "createdAt",
      "updatedAt",
    ]);

    const results = memories.map((mem) => {
      const metadata = Object.entries(mem.payload)
        .filter(([key]) => !excludedKeys.has(key))
        .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
      const tagNames = mem.payload.tagNames || [];
      const memory = TagsUtils.clarifyData(mem.payload.data, tagNames);
      return {
        id: mem.id,
        memory,
        hash: mem.payload.hash,
        createdAt: mem.payload.createdAt,
        updatedAt: mem.payload.updatedAt,
        metadata,
        ...(mem.payload.userId && { userId: mem.payload.userId }),
        ...(mem.payload.agentId && { agentId: mem.payload.agentId }),
        ...(mem.payload.runId && { runId: mem.payload.runId }),
      };
    });

    return { results, relations: [] };
  }
}
