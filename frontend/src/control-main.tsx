import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import BaseControlPage from './pages/BaseControlPage.tsx';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BaseControlPage />
  </StrictMode>,
);
