import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ICO Aggregator",
  description:
    "Enter a VIN once, receive real-time cash offers from every major car-buying site.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
