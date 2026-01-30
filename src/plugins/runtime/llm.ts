import { completeSimple, type TextContent } from "@mariozechner/pi-ai";

import type { MoltbotConfig } from "../../config/config.js";
import { getApiKeyForModel, requireApiKey } from "../../agents/model-auth.js";
import { resolveDefaultModelForAgent, modelKey } from "../../agents/model-selection.js";
import { resolveModel } from "../../agents/pi-embedded-runner/model.js";
import type { PluginLlmCompleteOptions, PluginLlmCompleteResult } from "./types.js";

function isTextContentBlock(block: unknown): block is TextContent {
  return typeof block === "object" && block !== null && (block as TextContent).type === "text";
}

/**
 * Create an LLM complete function bound to the given config.
 */
export function createPluginLlmComplete(cfg: MoltbotConfig) {
  return async (
    prompt: string,
    options?: PluginLlmCompleteOptions,
  ): Promise<PluginLlmCompleteResult> => {
    const modelRef = options?.model
      ? parseModelOverride(options.model, cfg)
      : resolveDefaultModelForAgent({ cfg });

    const { model, error } = resolveModel(
      modelRef.provider,
      modelRef.model,
      undefined,
      cfg,
    );

    if (!model || error) {
      throw new Error(`Could not resolve model ${modelRef.provider}/${modelRef.model}: ${error}`);
    }

    const auth = await getApiKeyForModel({ model, cfg });
    const apiKey = requireApiKey(auth, model.provider);

    const now = Date.now();
    const userContent = options?.systemPrompt
      ? `${options.systemPrompt}\n\n${prompt}`
      : prompt;
    const messages = [{ role: "user" as const, content: userContent, timestamp: now }];

    const response = await completeSimple(
      model,
      { messages },
      { apiKey, maxTokens: options?.maxTokens },
    );

    const text = response.content
      .filter(isTextContentBlock)
      .map((block: TextContent) => block.text.trim())
      .filter(Boolean)
      .join(" ")
      .trim();

    return {
      text,
      model: modelKey(modelRef.provider, modelRef.model),
    };
  };
}

/**
 * Check if LLM is available (model and auth configured).
 */
export function createPluginLlmIsAvailable(cfg: MoltbotConfig) {
  return async (): Promise<boolean> => {
    try {
      const modelRef = resolveDefaultModelForAgent({ cfg });
      const { model, error } = resolveModel(
        modelRef.provider,
        modelRef.model,
        undefined,
        cfg,
      );

      if (!model || error) return false;

      const auth = await getApiKeyForModel({ model, cfg });
      return Boolean(auth.apiKey?.trim());
    } catch {
      return false;
    }
  };
}

/**
 * Get the currently configured primary model.
 */
export function createPluginLlmGetModel(cfg: MoltbotConfig) {
  return (): string => {
    const modelRef = resolveDefaultModelForAgent({ cfg });
    return modelKey(modelRef.provider, modelRef.model);
  };
}

function parseModelOverride(
  raw: string,
  cfg: MoltbotConfig,
): { provider: string; model: string } {
  const trimmed = raw.trim();
  const slash = trimmed.indexOf("/");
  if (slash === -1) {
    const defaultRef = resolveDefaultModelForAgent({ cfg });
    return { provider: defaultRef.provider, model: trimmed };
  }
  return {
    provider: trimmed.slice(0, slash).trim(),
    model: trimmed.slice(slash + 1).trim(),
  };
}
