/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverComponentsExternalPackages: ["@browserbasehq/stagehand", "playwright", "playwright-core"],
  },
};

module.exports = nextConfig;
