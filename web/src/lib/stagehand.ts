import { Stagehand, type ConstructorParams } from "@browserbasehq/stagehand";

type StagehandModel = NonNullable<ConstructorParams["modelName"]>;

/**
 * Build a Stagehand session. Uses Browserbase for stealth when credentials are
 * present, falls back to a local Playwright Chromium otherwise.
 *
 * Callers MUST `await session.close()` (or `finally`) — browser resources leak otherwise.
 */
export async function createStagehand(): Promise<Stagehand> {
  const env =
    (process.env.STAGEHAND_ENV === "BROWSERBASE" ||
      (process.env.BROWSERBASE_API_KEY && process.env.STAGEHAND_ENV !== "LOCAL"))
      ? "BROWSERBASE"
      : "LOCAL";

  const stagehand = new Stagehand({
    env,
    apiKey: process.env.BROWSERBASE_API_KEY,
    projectId: process.env.BROWSERBASE_PROJECT_ID,
    modelName: (process.env.STAGEHAND_MODEL as StagehandModel | undefined) ?? "gpt-4o-mini",
    modelClientOptions: {
      apiKey: process.env.OPENAI_API_KEY ?? process.env.ANTHROPIC_API_KEY,
    },
    verbose: 1,
  });

  await stagehand.init();
  return stagehand;
}
