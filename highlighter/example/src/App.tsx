import React, { useState, useEffect, useCallback, useRef } from "react";

import {
  AreaHighlight,
  Highlight,
  PdfHighlighter,
  PdfLoader,
  Popup,
  Tip,
} from "./react-pdf-highlighter";
import type {
  Content,
  IHighlight,
  NewHighlight,
  ScaledPosition,
} from "./react-pdf-highlighter";

import { Sidebar } from "./Sidebar";
import { Spinner } from "./Spinner";
import { testHighlights as _testHighlights } from "./test-highlights";
import { ErrorFallback } from './ErrorFallback';

import "./style/App.css";
import "../../dist/style.css";


const getNextId = () => String(Math.random()).slice(2);

const parseIdFromHash = () =>
  document.location.hash.slice("#highlight-".length);

const resetHash = () => {
  document.location.hash = "";
};

const HighlightPopup = ({
  comment,
}: {
  comment: { text: string; emoji: string };
}) =>
  comment.text ? (
    <div className="Highlight__popup">
      {comment.emoji} {comment.text}
    </div>
  ) : null;

export default function App() {
  const searchParams = new URLSearchParams(document.location.search);
  const userId = searchParams.get("userId") || "";
  const documentsStr = searchParams.get("documents") || "{}";
  const documents = JSON.parse(decodeURIComponent(documentsStr));

  // Get first document and its highlights
  const firstPdfId = Object.keys(documents)[0];
  const firstHighlightIds = documents[firstPdfId] || [];

  console.log('Initial data:', {
    userId,
    firstPdfId,
    firstHighlightIds,
    documents
  });

  const [url, setUrl] = useState("");
  const [highlights, setHighlights] = useState<Array<IHighlight>>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [highlightsError, setHighlightsError] = useState<string | null>(null);

  const resetHighlights = () => {
    setHighlights([]);
  };

  const scrollViewerTo = useRef((highlight: IHighlight) => {});

  const getHighlightById = useCallback((id: string) => {
    return highlights.find((highlight) => highlight.id === id);
  }, [highlights]);

  const scrollToHighlightFromHash = useCallback(() => {
    const highlightId = parseIdFromHash();
    const highlight = getHighlightById(highlightId);
    
    if (highlight) {
      scrollViewerTo.current(highlight);
    }
  }, [getHighlightById]);

  useEffect(() => {
    window.addEventListener("hashchange", scrollToHighlightFromHash, false);
    return () => {
      window.removeEventListener(
        "hashchange",
        scrollToHighlightFromHash,
        false,
      );
    };
  }, [scrollToHighlightFromHash]);

  const updateHighlight = (
    highlightId: string,
    position: Partial<ScaledPosition>,
    content: Partial<Content>,
  ) => {
    console.log("Updating highlight", highlightId, position, content);
    setHighlights((prevHighlights) =>
      prevHighlights.map((h) => {
        const {
          id,
          position: originalPosition,
          content: originalContent,
          ...rest
        } = h;
        return id === highlightId
          ? {
              id,
              position: { ...originalPosition, ...position },
              content: { ...originalContent, ...content },
              ...rest,
            }
          : h;
      }),
    );
  };

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setPdfError(null);
      setHighlightsError(null);

      if (!firstPdfId || !userId) {
        console.error('Missing required parameters');
        setPdfError('No document selected');
        setIsLoading(false);
        return;
      }

      try {
        // Set PDF URL
        const pdfUrl = `http://localhost:5173/pdf/${firstPdfId}?user_id=${userId}`;
        console.log('PDF URL:', pdfUrl);
        setUrl(pdfUrl);

        // Fetch highlights
        console.log('Fetching highlights...');
        try {
          const highlightsResponse = await fetch('http://localhost:5173/document-highlights', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              pdf_id: parseInt(firstPdfId),
              highlight_ids: firstHighlightIds
            })
          });

          if (!highlightsResponse.ok) {
            throw new Error(`Failed to fetch highlights: ${highlightsResponse.statusText}`);
          }
          
          const highlightsData = await highlightsResponse.json();
          console.log('Received highlights data:', highlightsData);
          
          const transformedHighlights = highlightsData.highlights.map((h: any) => ({
            id: h.highlight_id,
            position: h.position,
            content: {
              text: h.content_text
            },
            comment: {
              text: h.comment_text,
              emoji: h.comment_emoji
            }
          }));

          setHighlights(transformedHighlights);
        } catch (error) {
          console.error('Error fetching highlights:', error);
          setHighlightsError('Unable to load highlights');
          // Still set empty highlights to allow PDF viewing
          setHighlights([]);
        }
      } catch (error) {
        console.error('Error in data fetching:', error);
        setPdfError('Unable to load document');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [firstPdfId, userId]);

  return (
    <div className="App" style={{ display: "flex", height: "100vh" }}>
      <Sidebar
        highlights={highlights}
        resetHighlights={resetHighlights}
        isLoading={isLoading}
      />
      <div
        style={{
          height: "100vh",
          width: "75vw",
          position: "relative",
        }}
      >
        {highlightsError && (
          <div style={{ 
            position: 'absolute', 
            top: 10, 
            right: 10, 
            background: '#fff3e0', 
            color: '#ef6c00', 
            padding: '8px 12px', 
            borderRadius: '4px',
            fontSize: '14px',
            zIndex: 1000,
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            {highlightsError}
          </div>
        )}
        
        {isLoading ? (
          <Spinner />
        ) : pdfError ? (
          <ErrorFallback 
            message="Document Unavailable"
            details={pdfError}
          />
        ) : (
          <PdfLoader 
            url={url} 
            beforeLoad={<Spinner />}
            errorMessage={
              <ErrorFallback 
                message="Unable to load PDF"
                details="The document could not be loaded. Please try again later."
              />
            }
          >
            {(pdfDocument) => (
              <PdfHighlighter
                pdfDocument={pdfDocument}
                enableAreaSelection={(event) => event.altKey}
                onScrollChange={resetHash}
                scrollRef={(scrollTo) => {
                  scrollViewerTo.current = scrollTo;
                  scrollToHighlightFromHash();
                }}
                onSelectionFinished={(
                  position,
                  content,
                  hideTipAndSelection,
                  transformSelection,
                ) => {
                  return null;
                }}
                highlightTransform={(
                  highlight,
                  index,
                  setTip,
                  hideTip,
                  viewportToScaled,
                  screenshot,
                  isScrolledTo,
                ) => {
                  const isTextHighlight = !highlight.content?.image;

                  const component = isTextHighlight ? (
                    <Highlight
                      isScrolledTo={isScrolledTo}
                      position={highlight.position}
                      comment={highlight.comment}
                    />
                  ) : (
                    <AreaHighlight
                      isScrolledTo={isScrolledTo}
                      highlight={highlight}
                      onChange={(boundingRect) => {
                        updateHighlight(
                          highlight.id,
                          { boundingRect: viewportToScaled(boundingRect) },
                          { image: screenshot(boundingRect) },
                        );
                      }}
                    />
                  );

                  return (
                    <Popup
                      popupContent={<HighlightPopup {...highlight} />}
                      onMouseOver={(popupContent) =>
                        setTip(highlight, (highlight) => popupContent)
                      }
                      onMouseOut={hideTip}
                      key={index}
                    >
                      {component}
                    </Popup>
                  );
                }}
                highlights={highlights}
              />
            )}
          </PdfLoader>
        )}
      </div>
    </div>
  );
}
