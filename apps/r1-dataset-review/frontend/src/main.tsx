import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConfigProvider, theme } from 'antd';
import 'antd/dist/reset.css';
import './styles.css';
import App from './App';
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><ConfigProvider theme={{algorithm:theme.darkAlgorithm,token:{colorPrimary:'#63c69b',colorBgContainer:'#17201d',borderRadius:6,fontSize:16}}}><App/></ConfigProvider></React.StrictMode>);
