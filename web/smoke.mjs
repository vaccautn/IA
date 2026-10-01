import { chromium } from "playwright-core";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
const browser = await chromium.launch({ channel: "msedge", headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("http://127.0.0.1:8010");
  await page.getByRole("heading", { name: "Detecci\xF3n en video" }).waitFor();
  await page.locator(".viewer img").waitFor({ timeout: 15e3 });
  await page.evaluate(() => document.fonts.ready);
  assert.ok(await page.evaluate(() => document.fonts.check("16px Manrope")), "Manrope must load");
  await page.locator(".viewer img").evaluate((image) => image.decode());
  const original = await page.locator(".viewer img").getAttribute("src");
  await page.getByRole("slider").fill("10");
  await page.waitForFunction((previous) => document.querySelector(".viewer img")?.src && !document.querySelector(".viewer img").src.endsWith(previous), original);
  await page.locator(".viewer img").evaluate((image) => image.decode());
  assert.match(await page.locator(".viewer img").getAttribute("alt"), /2\.00 segundos/);
  await page.getByRole("button", { name: "Reproducir frames" }).click();
  await page.getByRole("button", { name: "Pausar" }).click();
  await page.locator("input[type=file]").setInputFiles({ name: "incorrecto.txt", mimeType: "text/plain", buffer: Buffer.from("not video") });
  await page.getByRole("alert").waitFor();
  assert.match(await page.getByRole("alert").textContent(), /MP4/);
  await page.getByRole("button", { name: "Cerrar aviso" }).click();
  await mkdir("../outputs", { recursive: true });
  await page.screenshot({ path: "../outputs/video-web-desktop.png", fullPage: true });
  await page.reload();
  await page.locator(".viewer img").waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "../outputs/video-web-mobile.png", fullPage: true });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Mobile layout must not overflow");
  assert.deepEqual(errors, []);
  console.log("PASS: font, results, seeking, playback, invalid upload, reload, mobile layout, no JS exceptions.");
} finally {
  await browser.close();
}
