import { BaseScraper } from "./base";
import { CarMaxScraper } from "./carmax";

/**
 * Providers the aggregator fans out to. Phase 1 ships CarMax only; additional
 * providers (Carvana, Vroom, Peddle, KBB, Edmunds, TrueCar, CarGurus, Cars.com,
 * Driveway, AutoNation, EchoPark, Wheelzy, CarEdge) register themselves here.
 */
export const providers: BaseScraper[] = [new CarMaxScraper()];

export function getProvider(id: string): BaseScraper | undefined {
  return providers.find((p) => p.meta.id === id);
}
