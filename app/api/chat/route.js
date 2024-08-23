import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { GoogleGenerativeAI } from "@google/generative-ai";

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed. Do Not Use Markdown.
`

export async function POST(req) {
    const data = await req.json();
    const text = data[data.length - 1].content

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')

    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
    const model = genAI.getGenerativeModel({ model: "text-embedding-004" });
    const result = await model.embedContent(text);
    const embedding = result.embedding;
    // console.log(embedding.values)
    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.values,
    })

    let resultString = ''
    results.matches.forEach((match) => {
        resultString +=
            `
Returned Results:
Professor: ${match.id}
Review: ${match.metadata.stars}
Subject: ${match.metadata.subject}
Stars: ${match.metadata.stars}
  \n\n`
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const completion = await model.generateContentStream({
        prompt: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent },
        ].map(msg => msg.content).join("\n"),
        stream: true,
    });

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion.stream) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })
    return new NextResponse(stream)
}