import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Zynaptrix | Industrial AI Copilot",
  description: "Real-time Multi-Asset Factory Fleet Monitoring & Diagnostic Platform",
};

import { ReduxProvider } from "../store/Provider";
import NavBar from "../components/NavBar";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-slate-950">
        <ReduxProvider>
          <NavBar />
          <main className="flex-1">
            {children}
          </main>
        </ReduxProvider>
      </body>
    </html>
  );
}
